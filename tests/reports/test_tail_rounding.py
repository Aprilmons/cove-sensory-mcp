"""Preserve valid tail evidence when a Provider rounds the endpoint slightly up."""

from __future__ import annotations

import json

import pytest

from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.reports.normalize import normalize_provider_text


def _normalize_item(field, start, end, duration):
    payload = {
        "observations": [
            {
                "modality": "audio",
                "summary": "One audible phrase.",
                "segments": [],
                "transcript": [],
                "warnings": [],
                "confidence": "high",
                field: [
                    {
                        "start_seconds": start,
                        "end_seconds": end,
                        "text": "Complete phrase: seven four two.",
                    }
                ],
            }
        ]
    }
    return normalize_provider_text(
        json.dumps(payload),
        expected_modalities=frozenset({Modality.AUDIO}),
        duration_seconds=duration,
    ).observations[0]


@pytest.mark.parametrize("field", ["segments", "transcript"])
@pytest.mark.parametrize("end", [4.768, 4.77, 4.773])
def test_rounding_within_five_milliseconds_keeps_complete_tail(field, end):
    observation = _normalize_item(field, 2.0, end, 4.768)
    retained = getattr(observation, field)
    assert len(retained) == 1
    assert retained[0].text == "Complete phrase: seven four two."
    assert retained[0].end_seconds == 4.768
    assert observation.warnings == []


def test_exact_five_millisecond_tolerance_avoids_subtraction_precision_error():
    observation = _normalize_item("transcript", 0.0, 0.305, 0.3)
    assert observation.transcript[0].end_seconds == 0.3
    assert observation.warnings == []


@pytest.mark.parametrize("field", ["segments", "transcript"])
@pytest.mark.parametrize("end", [4.7731, 4.774, 8.0])
def test_larger_tail_overruns_are_still_omitted_with_existing_warning(field, end):
    observation = _normalize_item(field, 2.0, end, 4.768)
    assert getattr(observation, field) == []
    assert [(warning.code, warning.message) for warning in observation.warnings] == [
        (
            "TIMECODE_OUT_OF_RANGE",
            "One or more timecoded items were omitted because they fell outside the media range.",
        )
    ]


@pytest.mark.parametrize("end", [4.7684, 4.7686, 4.77])
def test_schema_rounding_cannot_push_retained_endpoint_past_fractional_duration(end):
    observation = _normalize_item("transcript", 2.0, end, 4.7686)
    assert observation.transcript[0].end_seconds == 4.768
    assert observation.transcript[0].end_seconds <= 4.7686
    assert observation.warnings == []


@pytest.mark.parametrize(
    "start,end,duration",
    [
        (-0.0004, 4.77, 4.768),
        (4.77, 4.77, 4.768),
        (4.771, 4.77, 4.768),
        (4.7681, 4.77, 4.768),
        (4.7679, 4.77, 4.768),
        (0.0, 0.0006, 0.0006),
        (0.0, 0.005, 0.0),
    ],
)
def test_invalid_or_collapsed_intervals_are_not_rescued_by_tail_tolerance(
    start, end, duration
):
    observation = _normalize_item("transcript", start, end, duration)
    assert observation.transcript == []
    assert [warning.code for warning in observation.warnings] == [
        "TIMECODE_OUT_OF_RANGE"
    ]


def test_unknown_duration_preserves_existing_rounding_behavior():
    observation = _normalize_item("transcript", 0.0004, 4.7686, None)
    assert observation.transcript[0].start_seconds == 0.0
    assert observation.transcript[0].end_seconds == 4.769
    assert observation.warnings == []
