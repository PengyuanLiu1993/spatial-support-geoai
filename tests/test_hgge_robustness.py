from __future__ import annotations

import numpy as np

from scripts.summarise_hgge_robustness import block_bootstrap_rmse_difference


def test_block_bootstrap_detects_worse_ablated_prediction() -> None:
    observed = np.linspace(0.0, 10.0, 200)
    full = observed + 0.1
    ablated = observed + 1.0
    blocks = np.repeat(np.arange(20), 10)
    result = block_bootstrap_rmse_difference(
        observed, full, ablated, blocks, replicates=500, seed=83
    )
    assert result["ablated_minus_full_rmse"] > 0.8
    assert result["bootstrap_lower"] > 0.0
