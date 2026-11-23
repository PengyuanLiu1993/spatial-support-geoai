import torch

from gravity_gnn.data import (
    assert_group_disjoint,
    make_fold_split,
    make_group_split,
    make_synthetic_graph,
    strict_partition_edge_mask,
    validate_graph,
)
from gravity_gnn.training import grouped_mse_loss, standardize_building_features


def test_synthetic_graph_is_valid() -> None:
    data = make_synthetic_graph(num_buildings=96, num_amenities=24, seed=7)
    validate_graph(data)
    assert data.num_nodes == 120
    assert data.num_relations == 2
    assert torch.isnan(data.y[~data.building_mask]).all()


def test_group_split_is_disjoint_and_edges_do_not_cross() -> None:
    data = make_synthetic_graph(num_buildings=160, num_amenities=24, seed=11)
    split = make_group_split(data, seed=11)
    assert_group_disjoint(data, split)
    assert (split.train | split.validation | split.test).sum() == data.building_mask.sum()

    edge_mask = strict_partition_edge_mask(data, split.partition)
    source, destination = data.edge_index[:, edge_mask]
    source_partition = split.partition[source]
    destination_partition = split.partition[destination]
    assert not torch.any(
        (source_partition >= 0)
        & (destination_partition >= 0)
        & (source_partition != destination_partition)
    )


def test_fixed_fold_split_keeps_complete_groups_together() -> None:
    data = make_synthetic_graph(num_buildings=160, num_amenities=24, seed=19)
    groups = torch.unique(data.group[data.building_mask])
    group_to_fold = torch.full((int(groups.max()) + 1,), -1, dtype=torch.long)
    group_to_fold[groups] = torch.arange(len(groups)) % 5
    split = make_fold_split(data, group_to_fold, test_fold=0, validation_fold=1)
    assert_group_disjoint(data, split)
    assert (split.train | split.validation | split.test).sum() == data.building_mask.sum()
    assert set(group_to_fold[data.group[split.test]].tolist()) == {0}
    assert set(group_to_fold[data.group[split.validation]].tolist()) == {1}


def test_feature_scaling_preserves_unlabelled_context_encodings() -> None:
    data = make_synthetic_graph(num_buildings=96, num_amenities=24, seed=23)
    split = make_group_split(data, seed=23)
    standardized, mean, scale = standardize_building_features(data, split.train)
    assert torch.equal(standardized[~data.building_mask], data.x[~data.building_mask])
    assert torch.isfinite(standardized).all()
    assert torch.isfinite(mean).all()
    assert torch.all(scale > 0)


def test_grouped_loss_supervises_group_means_not_individual_nodes() -> None:
    target = torch.tensor([2.0, 2.0, 5.0, 5.0])
    prediction = torch.tensor([1.0, 3.0, 4.0, 6.0])
    group = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(4, dtype=torch.bool)
    assert grouped_mse_loss(target, prediction, group, mask).item() == 0.0
