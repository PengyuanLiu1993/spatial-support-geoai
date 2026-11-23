from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass
class GraphData:
    """Tensor-only graph bundle used by all models.

    Nodes share a padded feature space. Edges are directed; for a message into
    building ``v``, ``edge_index[0]`` is the source and ``edge_index[1]`` is
    the destination.
    """

    x: Tensor
    node_mass: Tensor
    edge_index: Tensor
    edge_impedance: Tensor
    edge_type: Tensor
    y: Tensor
    building_mask: Tensor
    group: Tensor
    label_group: Tensor | None = None
    sample_weight: Tensor | None = None
    feature_names: tuple[str, ...] | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_relations(self) -> int:
        return int(self.edge_type.max().item()) + 1

    def to(self, device: torch.device | str) -> "GraphData":
        values: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, Tensor) else value
        return GraphData(**values)

    def with_edge_mask(self, mask: Tensor) -> "GraphData":
        if mask.dtype != torch.bool or mask.numel() != self.num_edges:
            raise ValueError("edge mask must be boolean with one value per edge")
        return GraphData(
            x=self.x,
            node_mass=self.node_mass,
            edge_index=self.edge_index[:, mask],
            edge_impedance=self.edge_impedance[mask],
            edge_type=self.edge_type[mask],
            y=self.y,
            building_mask=self.building_mask,
            group=self.group,
            label_group=self.label_group,
            sample_weight=self.sample_weight,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True)
class GroupSplit:
    train: Tensor
    validation: Tensor
    test: Tensor
    partition: Tensor


def validate_graph(data: GraphData) -> None:
    n = data.num_nodes
    e = data.num_edges
    if data.x.ndim != 2 or data.x.shape[0] != n:
        raise ValueError("x must have shape [N, F]")
    if data.edge_index.shape != (2, e):
        raise ValueError("edge_index must have shape [2, E]")
    for name in ("node_mass", "y", "building_mask", "group"):
        if getattr(data, name).shape != (n,):
            raise ValueError(f"{name} must have shape [N]")
    for name in ("edge_impedance", "edge_type"):
        if getattr(data, name).shape != (e,):
            raise ValueError(f"{name} must have shape [E]")
    if data.edge_index.dtype != torch.long or data.edge_type.dtype != torch.long:
        raise ValueError("edge_index and edge_type must use int64")
    if data.building_mask.dtype != torch.bool:
        raise ValueError("building_mask must be boolean")
    if not torch.isfinite(data.x).all():
        raise ValueError("x contains non-finite values")
    if not torch.isfinite(data.node_mass).all() or torch.any(data.node_mass <= 0):
        raise ValueError("node_mass must be finite and strictly positive")
    if not torch.isfinite(data.edge_impedance).all() or torch.any(data.edge_impedance < 0):
        raise ValueError("edge_impedance must be finite and non-negative")
    if e == 0:
        raise ValueError("graph has no edges")
    if data.edge_index.min() < 0 or data.edge_index.max() >= n:
        raise ValueError("edge_index contains an out-of-range node id")
    relations = torch.unique(data.edge_type).cpu()
    expected = torch.arange(int(relations.max()) + 1)
    if not torch.equal(relations, expected):
        raise ValueError("edge types must be contiguous and start at zero")
    if not torch.isfinite(data.y[data.building_mask]).all():
        raise ValueError("every building must have a finite target")
    if torch.any(data.group[data.building_mask] < 0):
        raise ValueError("every labelled building must have a non-negative group")
    if data.label_group is not None:
        if data.label_group.shape != (n,) or data.label_group.dtype != torch.long:
            raise ValueError("label_group must use int64 with shape [N]")
        if torch.any(data.label_group[data.building_mask] < 0):
            raise ValueError("every labelled building must have a label group")
    if data.sample_weight is not None:
        if data.sample_weight.shape != (n,):
            raise ValueError("sample_weight must have shape [N]")
        if not torch.isfinite(data.sample_weight).all() or torch.any(
            data.sample_weight[data.building_mask] <= 0
        ):
            raise ValueError("building sample weights must be finite and positive")
    if data.feature_names is not None and len(data.feature_names) != data.num_features:
        raise ValueError("feature_names must contain one name per feature")


def make_group_split(
    data: GraphData,
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> GroupSplit:
    """Split complete spatial groups, never individual buildings."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("the test fraction must be positive")

    groups = torch.unique(data.group[data.building_mask])
    if len(groups) < 3:
        raise ValueError("at least three spatial groups are required")
    generator = torch.Generator().manual_seed(seed)
    groups = groups[torch.randperm(len(groups), generator=generator)]
    n_train = max(1, round(len(groups) * train_fraction))
    n_validation = max(1, round(len(groups) * validation_fraction))
    if n_train + n_validation >= len(groups):
        n_validation = 1
        n_train = len(groups) - 2

    train_groups = groups[:n_train]
    validation_groups = groups[n_train : n_train + n_validation]
    test_groups = groups[n_train + n_validation :]

    def membership(values: Tensor, choices: Tensor) -> Tensor:
        return (values[:, None] == choices[None, :]).any(dim=1)

    train = data.building_mask & membership(data.group, train_groups)
    validation = data.building_mask & membership(data.group, validation_groups)
    test = data.building_mask & membership(data.group, test_groups)
    partition = torch.full((data.num_nodes,), -1, dtype=torch.long)
    partition[train] = 0
    partition[validation] = 1
    partition[test] = 2
    return GroupSplit(train=train, validation=validation, test=test, partition=partition)


def make_fold_split(
    data: GraphData,
    group_to_fold: Tensor,
    test_fold: int,
    validation_fold: int,
) -> GroupSplit:
    """Create a train/validation/test split from a fixed spatial-fold plan.

    ``group_to_fold[g]`` gives the fold assigned to spatial group ``g``.  All
    labelled buildings in a group therefore remain together.  Unlabelled
    context nodes are left outside the partition and can only contribute via
    the graph edges retained by :func:`strict_partition_edge_mask`.
    """

    if test_fold == validation_fold:
        raise ValueError("test and validation folds must differ")
    if group_to_fold.ndim != 1 or group_to_fold.dtype != torch.long:
        raise ValueError("group_to_fold must be a one-dimensional int64 tensor")
    labelled_groups = data.group[data.building_mask]
    if labelled_groups.numel() == 0:
        raise ValueError("the graph has no labelled buildings")
    if labelled_groups.min() < 0 or labelled_groups.max() >= len(group_to_fold):
        raise ValueError("group_to_fold does not cover every labelled spatial group")
    labelled_folds = group_to_fold[labelled_groups]
    if torch.any(labelled_folds < 0):
        raise ValueError("every labelled spatial group must have a fold assignment")
    available = set(torch.unique(labelled_folds).cpu().tolist())
    if test_fold not in available or validation_fold not in available:
        raise ValueError("test and validation folds must exist in the fold plan")

    train = data.building_mask.clone()
    validation = torch.zeros(data.num_nodes, dtype=torch.bool)
    test = torch.zeros(data.num_nodes, dtype=torch.bool)
    validation[data.building_mask] = labelled_folds == validation_fold
    test[data.building_mask] = labelled_folds == test_fold
    train &= ~(validation | test)
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("every partition must contain at least one labelled building")

    partition = torch.full((data.num_nodes,), -1, dtype=torch.long)
    partition[train] = 0
    partition[validation] = 1
    partition[test] = 2
    return GroupSplit(train=train, validation=validation, test=test, partition=partition)


def strict_partition_edge_mask(data: GraphData, partition: Tensor) -> Tensor:
    """Remove graph links between labelled building partitions.

    Unlabelled context nodes have partition ``-1`` and remain available to all
    folds. This prevents building features from crossing train/validation/test
    boundaries while retaining shared amenity context.
    """

    if partition.shape != (data.num_nodes,):
        raise ValueError("partition must have shape [N]")
    source, destination = data.edge_index
    source_partition = partition[source]
    destination_partition = partition[destination]
    crosses = (
        (source_partition >= 0)
        & (destination_partition >= 0)
        & (source_partition != destination_partition)
    )
    return ~crosses


def assert_group_disjoint(data: GraphData, split: GroupSplit) -> None:
    group_sets = []
    for mask in (split.train, split.validation, split.test):
        group_sets.append(set(data.group[mask].cpu().tolist()))
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise AssertionError("a spatial group appears in more than one partition")


def load_graph_npz(path: str | Path) -> GraphData:
    source = np.load(Path(path), allow_pickle=False)
    required = {
        "x",
        "node_mass",
        "edge_index",
        "edge_impedance",
        "edge_type",
        "y",
        "building_mask",
        "group",
    }
    missing = required - set(source.files)
    if missing:
        raise ValueError(f"graph bundle is missing arrays: {sorted(missing)}")
    data = GraphData(
        x=torch.as_tensor(source["x"], dtype=torch.float32),
        node_mass=torch.as_tensor(source["node_mass"], dtype=torch.float32),
        edge_index=torch.as_tensor(source["edge_index"], dtype=torch.long),
        edge_impedance=torch.as_tensor(source["edge_impedance"], dtype=torch.float32),
        edge_type=torch.as_tensor(source["edge_type"], dtype=torch.long),
        y=torch.as_tensor(source["y"], dtype=torch.float32),
        building_mask=torch.as_tensor(source["building_mask"], dtype=torch.bool),
        group=torch.as_tensor(source["group"], dtype=torch.long),
        label_group=(
            torch.as_tensor(source["label_group"], dtype=torch.long)
            if "label_group" in source.files
            else None
        ),
        sample_weight=(
            torch.as_tensor(source["sample_weight"], dtype=torch.float32)
            if "sample_weight" in source.files
            else None
        ),
        feature_names=(
            tuple(source["feature_names"].astype(str).tolist())
            if "feature_names" in source.files
            else None
        ),
    )
    validate_graph(data)
    return data


def make_synthetic_graph(
    num_buildings: int = 480,
    num_amenities: int = 80,
    neighbors_per_building: int = 5,
    amenities_per_building: int = 10,
    seed: int = 20260826,
) -> GraphData:
    """Create a small heterogeneous graph with a known gravity signal.

    Coordinates are expressed in synthetic travel-time minutes. Relation 0 is
    amenity-to-building accessibility and relation 1 is symmetric local
    building proximity. The function is a code test, not empirical evidence.
    """

    if min(num_buildings, num_amenities) < 8:
        raise ValueError("synthetic graph requires at least eight nodes per type")
    generator = torch.Generator().manual_seed(seed)
    building_xy = 60.0 * torch.rand((num_buildings, 2), generator=generator)
    amenity_xy = 60.0 * torch.rand((num_amenities, 2), generator=generator)

    building_mass = torch.exp(0.7 * torch.randn(num_buildings, generator=generator))
    amenity_mass = torch.exp(0.9 * torch.randn(num_amenities, generator=generator))
    all_mass = torch.cat((building_mass, amenity_mass))
    all_mass = all_mass / all_mass.median()
    building_mass = all_mass[:num_buildings]
    amenity_mass = all_mass[num_buildings:]

    feature_count = 10
    x = torch.zeros((num_buildings + num_amenities, feature_count))
    compactness = 0.35 + 0.6 * torch.rand(num_buildings, generator=generator)
    elongation = 1.0 + 2.0 * torch.rand(num_buildings, generator=generator)
    height = 1.0 + torch.poisson(3.0 * torch.ones(num_buildings), generator=generator)
    x[:num_buildings, 0] = torch.log(building_mass)
    x[:num_buildings, 1] = compactness
    x[:num_buildings, 2] = elongation
    x[:num_buildings, 3] = height / 10.0
    x[:num_buildings, 4:6] = (building_xy - 30.0) / 30.0

    amenity_type = torch.randint(0, 4, (num_amenities,), generator=generator)
    x[num_buildings:, 0] = torch.log(amenity_mass)
    x[num_buildings:, 6:10] = torch.nn.functional.one_hot(amenity_type, 4).float()

    building_to_amenity = torch.cdist(building_xy, amenity_xy)
    amenity_k = min(amenities_per_building, num_amenities)
    amenity_distance, amenity_index = torch.topk(
        building_to_amenity, k=amenity_k, dim=1, largest=False
    )
    accessibility_source = (amenity_index + num_buildings).reshape(-1)
    accessibility_destination = (
        torch.arange(num_buildings)[:, None].expand(-1, amenity_k).reshape(-1)
    )

    building_distance = torch.cdist(building_xy, building_xy)
    building_distance.fill_diagonal_(float("inf"))
    building_k = min(neighbors_per_building, num_buildings - 1)
    neighbor_distance, neighbor_index = torch.topk(
        building_distance, k=building_k, dim=1, largest=False
    )
    proximity_source = neighbor_index.reshape(-1)
    proximity_destination = (
        torch.arange(num_buildings)[:, None].expand(-1, building_k).reshape(-1)
    )

    edge_index = torch.stack(
        (
            torch.cat((accessibility_source, proximity_source)),
            torch.cat((accessibility_destination, proximity_destination)),
        )
    ).long()
    edge_impedance = torch.cat((amenity_distance.reshape(-1), neighbor_distance.reshape(-1)))
    edge_type = torch.cat(
        (
            torch.zeros(accessibility_source.numel(), dtype=torch.long),
            torch.ones(proximity_source.numel(), dtype=torch.long),
        )
    )

    type_effect = torch.tensor([1.3, 0.9, -0.7, 0.5])
    selected_type_effect = type_effect[amenity_type[amenity_index]].reshape(-1)
    selected_mass = amenity_mass[amenity_index].reshape(-1)
    selected_distance = amenity_distance.reshape(-1)
    true_weight = selected_mass.pow(0.22) * torch.exp(-((selected_distance / 11.5).pow(1.8)))
    numerator = torch.zeros(num_buildings).index_add_(
        0, accessibility_destination, true_weight * selected_type_effect
    )
    denominator = torch.zeros(num_buildings).index_add_(
        0, accessibility_destination, true_weight
    )
    accessibility_signal = numerator / denominator.clamp_min(1e-6)
    spatial_signal = 0.35 * torch.sin(building_xy[:, 0] / 8.0) + 0.25 * torch.cos(
        building_xy[:, 1] / 10.0
    )
    noise = 0.25 * torch.randn(num_buildings, generator=generator)
    target = (
        28.0
        + 7.0 * accessibility_signal
        + 1.8 * torch.log(building_mass)
        + 1.2 * compactness
        + 2.0 * spatial_signal
        + noise
    ).clamp_min(1.0)
    y = torch.full((num_buildings + num_amenities,), float("nan"))
    y[:num_buildings] = target

    grid_x = torch.clamp((building_xy[:, 0] / 10.0).long(), max=5)
    grid_y = torch.clamp((building_xy[:, 1] / 10.0).long(), max=5)
    group = torch.full((num_buildings + num_amenities,), -1, dtype=torch.long)
    group[:num_buildings] = grid_y * 6 + grid_x
    building_mask = torch.zeros(num_buildings + num_amenities, dtype=torch.bool)
    building_mask[:num_buildings] = True

    data = GraphData(
        x=x,
        node_mass=all_mass,
        edge_index=edge_index,
        edge_impedance=edge_impedance,
        edge_type=edge_type,
        y=y,
        building_mask=building_mask,
        group=group,
        label_group=group.clone(),
        sample_weight=torch.where(
            building_mask,
            torch.ones_like(y),
            torch.zeros_like(y),
        ),
    )
    validate_graph(data)
    return data
