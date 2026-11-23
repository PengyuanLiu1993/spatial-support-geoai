import numpy as np
import torch

from gravity_gnn.data import make_synthetic_graph
from gravity_gnn.models import HeterogeneousFacilityGravityGNN
from gravity_gnn.training import seed_everything
from scripts.run_heterogeneous_facility_gnn import (
    exclude_relation_edges,
    parse_relation_access_shifts,
    relation_access_counterfactuals,
)


def test_relation_access_shift_parser_requires_ordered_zero_origin() -> None:
    assert parse_relation_access_shifts("0,5,10") == (0.0, 5.0, 10.0)
    for invalid in ("5,10", "0,10,5", "0,5,5", "0,45"):
        try:
            parse_relation_access_shifts(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid relation-access shifts: {invalid}")


def test_relation_exclusion_removes_only_named_edges() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=59)
    relation_names = [f"relation_{index}" for index in range(data.num_relations)]
    excluded = relation_names[0]
    filtered = exclude_relation_edges(data, relation_names, (excluded,))
    assert filtered.num_nodes == data.num_nodes
    assert filtered.num_edges < data.num_edges
    assert not torch.any(filtered.edge_type == 0)
    assert torch.equal(
        filtered.edge_type,
        data.edge_type[data.edge_type != 0],
    )


def test_relation_access_counterfactual_reaches_zero_at_reference() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=53)
    data.x[~data.building_mask] = 0.0
    seed_everything(53)
    model = HeterogeneousFacilityGravityGNN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=1,
        dropout=0.0,
        queen_message_passing=False,
    ).eval()
    labelled_nodes = np.flatnonzero(data.building_mask.numpy())
    access = relation_access_counterfactuals(
        model,
        data,
        train_mask=data.building_mask,
        target_node_limit=data.num_nodes,
        target_labelled_nodes=labelled_nodes,
        relation_count=data.num_relations - 1,
        shifts=(0.0, 40.0),
        residual_weight=1.0,
        device="cpu",
    )
    assert access.shape == (1, len(labelled_nodes), 2)
    assert np.isfinite(access).all()
    np.testing.assert_allclose(access[:, :, 1], 0.0, atol=1e-6)
