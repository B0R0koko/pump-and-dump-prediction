"""Tiny timestamp examples verify the compact per-asset audit's conclusions."""

from datetime import datetime

import pytest

from pumps_and_dumps.features.audit import AssetInputAudit
from pumps_and_dumps.features.provenance import timestamp_ns, utc_timestamp

CUTOFF = datetime(2021, 6, 1, 12)
BOUNDARY = timestamp_ns(CUTOFF)


def _record(role: str, timestamp: int | None, complete: bool = True) -> dict:
    return {
        "role": role,
        "data_start_ns": timestamp,
        "data_end_ns": timestamp,
        "provenance_complete": complete,
    }


def test_summary_unions_actual_inputs_across_features_and_target_windows() -> None:
    """All callbacks contribute to one maximum/minimum without storing records.

    Three predictors end ten, one, and three nanoseconds before the decision.
    Three target inputs begin ten, two, and five nanoseconds after it, including
    the earliest return baseline. Empty additional windows must not erase
    either extreme. The summary must therefore show -1 ns and +2 ns, with both
    checks true and nine fractional digits preserved in the readable output.
    """
    audit = AssetInputAudit()
    for offset in (-10, -1, -3):
        audit.observe(_record("predictor", BOUNDARY + offset))
    for offset in (10, 2, 5):
        audit.observe(_record("outcome", BOUNDARY + offset))
    audit.observe(_record("predictor", None))
    audit.observe(_record("outcome", None))

    assert audit.summary(CUTOFF) == {
        "feature_max_time": "2021-06-01T11:59:59.999999999Z",
        "target_min_time": "2021-06-01T12:00:00.000000002Z",
        "non_overlapping": True,
        "features_before_cutoff": True,
        "provenance_complete": True,
        "status": "OK",
    }


@pytest.mark.parametrize(
    "feature_offset,target_offset,status,non_overlapping,before_cutoff",
    [
        (-2, -1, "OK", True, True),
        (-1, -1, "OVERLAP", False, True),
        (-1, -2, "OVERLAP", False, True),
        (0, 1, "LEAKAGE", True, False),
        (0, 0, "LEAKAGE", False, False),
        (1, 2, "LEAKAGE", True, False),
        (-1, None, "NO_DATA", None, True),
        (None, 1, "NO_DATA", None, None),
        (None, None, "NO_DATA", None, None),
    ],
)
def test_summary_distinguishes_touching_bounds_cutoff_and_absent_data(
    feature_offset, target_offset, status, non_overlapping, before_cutoff
) -> None:
    """A one-nanosecond gap passes, but shared endpoints count as overlap.

    Offsets are relative to noon. The target may start before noon because its
    return uses a prior-price baseline; sharing that timestamp with a predictor
    is OVERLAP even when predictors remain causal. A predictor exactly at or
    after noon is LEAKAGE regardless of target separation. Missing either side
    is NO_DATA and must never be reported as successful non-overlap.
    """
    feature = None if feature_offset is None else BOUNDARY + feature_offset
    target = None if target_offset is None else BOUNDARY + target_offset
    audit = AssetInputAudit()
    audit.observe(_record("predictor", feature))
    audit.observe(_record("outcome", target))

    result = audit.summary(CUTOFF)

    assert result["status"] == status
    assert result["non_overlapping"] is non_overlapping
    assert result["features_before_cutoff"] is before_cutoff
    assert result["feature_max_time"] == utc_timestamp(feature)
    assert result["target_min_time"] == utc_timestamp(target)


@pytest.mark.parametrize(
    "feature_offset,target_offset,incomplete_role,status,non_overlapping,before_cutoff",
    [
        (-2, -1, "predictor", "UNKNOWN", None, None),
        (-2, -1, "outcome", "UNKNOWN", None, True),
        (-1, -1, "predictor", "OVERLAP", False, None),
        (-1, -1, "outcome", "OVERLAP", False, True),
        (0, 1, "predictor", "LEAKAGE", None, False),
        (0, 0, "outcome", "LEAKAGE", False, False),
        (None, None, "predictor", "UNKNOWN", None, None),
    ],
)
def test_incomplete_provenance_cannot_certify_safety_but_retains_known_violations(
    feature_offset, target_offset, incomplete_role, status, non_overlapping, before_cutoff
) -> None:
    """An unknown dependency may invalidate apparently separated known bounds.

    First observe fully traced inputs, then an additional dependency with a
    missing timestamp. Completeness must remain false and unsafe conclusions
    must not be erased: known touching bounds still prove OVERLAP, and a known
    predictor at the cutoff still proves LEAKAGE. A missing target dependency
    affects separation but does not invalidate independently complete predictor
    cutoff checking. Fully absent, untraceable data is UNKNOWN rather than an
    empty input falsely certified as NO_DATA.
    """
    audit = AssetInputAudit()
    audit.observe(_record("predictor", None if feature_offset is None else BOUNDARY + feature_offset))
    audit.observe(_record("outcome", None if target_offset is None else BOUNDARY + target_offset))
    audit.observe(_record(incomplete_role, None, complete=False))
    audit.observe(_record(incomplete_role, None))

    result = audit.summary(CUTOFF)

    assert result["provenance_complete"] is False
    assert result["status"] == status
    assert result["non_overlapping"] is non_overlapping
    assert result["features_before_cutoff"] is before_cutoff
