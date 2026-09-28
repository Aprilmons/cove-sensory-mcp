# Private Railway deployment for ChatGPT

This deployment keeps Cove off the public internet. ChatGPT reaches it through
OpenAI's Secure MCP Tunnel, and attached media is handed to the sensing tools by
temporary download URL.

## Architecture

The Railway project contains two private services:

1. `cove` — built from this repository's `Dockerfile`.
2. `cove-tunnel` — the pinned OpenAI image
   `ghcr.io/openai/tunnel-client:v0.0.12`.

Do not generate a public Railway domain for either service.

## Cove service

Attach a Railway Volume at `/data`. The volume stores only non-secret
configuration and capability-verification state; request workspaces are
temporary and are cleaned after each sensing job.

Set these Railway variables:

| Variable | Value | Secret |
| --- | --- | --- |
| `GEMINI_API_KEY` | A Gemini Developer API key | Yes |
| `COVE_GEMINI_MODEL` | `gemini-3.7-flash` | No |
| `COVE_DATA_DIR` | `/data` | No |
| `PORT` | `8000` | No |

Use `/healthz` as the health-check path. The MCP endpoint is `/mcp` on port
`8000` inside Railway's private network.

## Tunnel service

Deploy the exact Docker image `ghcr.io/openai/tunnel-client:v0.0.12` and set:

| Variable | Value | Secret |
| --- | --- | --- |
| `CONTROL_PLANE_API_KEY` | OpenAI tunnel runtime API key | Yes |
| `CONTROL_PLANE_TUNNEL_ID` | Tunnel ID from OpenAI Platform | Treat as private |
| `MCP_SERVER_URL` | `http://cove.railway.internal:8000/mcp` | No |
| `HEALTH_LISTEN_ADDR` | `:8080` | No |
| `LOG_LEVEL` | `info` | No |
| `LOG_FORMAT` | `json` | No |

The tunnel service needs no volume and no public domain.

## First connection

In ChatGPT developer mode, create a private app and choose **Tunnel** as its
connection type. Select the tunnel created for this deployment.

Before normal use, call `sensory_self_test` once with all five modalities:

- `image`
- `video_visual`
- `video_audio`
- `audio`
- `music`

The self-test sends tiny packaged samples to Gemini, consumes a small amount of
provider quota, and records only the verified capability state on `/data`.

After verification, attach an image, audio file, song, or video in ChatGPT and
ask it to use the matching `sense_*` tool. ChatGPT passes a temporary HTTPS URL;
Cove downloads the file into a bounded job directory, inspects or normalizes it,
sends the prepared media to the configured provider, then removes the job files.

## Secret handling

Never put either API key in Git, a chat message, a tool argument, or a public
Railway variable. Enter keys only in Railway's secret-variable controls. The
generated Cove YAML stores the environment-variable name `GEMINI_API_KEY`, not
the key value.
