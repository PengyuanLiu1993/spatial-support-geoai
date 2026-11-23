from __future__ import annotations

import numpy as np

from scripts.merge_hgge_seed_runs import merge_payloads


def payload(seed: int, offset: float) -> dict[str, np.ndarray]:
    relation_access = np.full((1, 2, 3, 2), offset, dtype=float)
    return {
        "lsoa_codes": np.asarray(["a", "b", "c"]),
        "observed": np.asarray([1.0, 2.0, 3.0]),
        "fold": np.asarray([0, 1, 2]),
        "seed": np.asarray([seed]),
        "seed_predictions": np.asarray([[1.0, 2.0, 3.0]]) + offset,
        "ensemble_prediction": np.asarray([1.0, 2.0, 3.0]) + offset,
        "relation_names": np.asarray(["gp", "pharmacy"]),
        "relation_access_shift_minutes": np.asarray([0.0, 5.0]),
        "seed_relation_access": relation_access,
        "ensemble_relation_access": relation_access[0],
    }


def test_merge_payloads_orders_and_averages_disjoint_seeds() -> None:
    merged = merge_payloads([payload(2, 2.0), payload(1, 0.0)])
    assert merged["seed"].tolist() == [1, 2]
    assert np.allclose(merged["ensemble_prediction"], [2.0, 3.0, 4.0])
    assert np.allclose(merged["ensemble_relation_access"], 1.0)


def test_merge_payloads_rejects_duplicate_seeds() -> None:
    try:
        merge_payloads([payload(1, 0.0), payload(1, 1.0)])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate seeds must fail")


def test_merge_payloads_supports_prediction_only_runs() -> None:
    first = payload(1, 0.0)
    second = payload(2, 2.0)
    for item in (first, second):
        for key in (
            "relation_names",
            "relation_access_shift_minutes",
            "seed_relation_access",
            "ensemble_relation_access",
        ):
            item.pop(key)
    merged = merge_payloads([second, first])
    assert merged["seed"].tolist() == [1, 2]
    assert np.allclose(merged["ensemble_prediction"], [2.0, 3.0, 4.0])
    assert "seed_relation_access" not in merged


def test_merge_payloads_supports_mixed_auxiliary_exports() -> None:
    relation_payload = payload(1, 0.0)
    prediction_payload = payload(2, 2.0)
    for key in (
        "relation_names",
        "relation_access_shift_minutes",
        "seed_relation_access",
        "ensemble_relation_access",
    ):
        prediction_payload.pop(key)
    merged = merge_payloads([relation_payload, prediction_payload])
    assert merged["seed"].tolist() == [1, 2]
    assert np.allclose(merged["ensemble_prediction"], [2.0, 3.0, 4.0])
    assert "seed_relation_access" not in merged
