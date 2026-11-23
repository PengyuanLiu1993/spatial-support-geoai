from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import GraphData


def _inverse_softplus(value: float, epsilon: float = 1e-6) -> float:
    adjusted = max(value - epsilon, epsilon)
    return math.log(math.expm1(adjusted))


def _inverse_bounded_sigmoid(value: float, lower: float, upper: float) -> float:
    if not lower < value < upper:
        raise ValueError("initial value must lie strictly inside its bounds")
    probability = (value - lower) / (upper - lower)
    return math.log(probability / (1.0 - probability))


class GravityKernel(nn.Module):
    """Positive, optionally relation-specific gravity-decay kernel.

    ``edge_impedance`` may use different units only when relation-specific
    parameters are enabled and those units are recorded in data provenance.
    Masses should be positive and median-scaled before entering the model.
    """

    def __init__(
        self,
        num_relations: int,
        relation_specific: bool = True,
        learnable: bool = True,
        mass_enabled: bool = True,
        initial_alpha: float = 0.15,
        initial_scale: float = 15.0,
        initial_beta: float = 1.5,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_relations < 1:
            raise ValueError("num_relations must be positive")
        if min(initial_alpha, initial_scale, initial_beta) <= 0:
            raise ValueError("kernel initial values must be positive")
        self.num_relations = num_relations
        self.relation_specific = relation_specific
        self.mass_enabled = mass_enabled
        self.epsilon = epsilon
        count = num_relations if relation_specific else 1
        self.raw_alpha = nn.Parameter(
            torch.full((count,), _inverse_softplus(initial_alpha, epsilon)),
            requires_grad=learnable and mass_enabled,
        )
        self.raw_scale = nn.Parameter(
            torch.full((count,), _inverse_softplus(initial_scale, epsilon)),
            requires_grad=learnable,
        )
        self.raw_beta = nn.Parameter(
            torch.full((count,), _inverse_softplus(initial_beta, epsilon)),
            requires_grad=learnable,
        )

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.raw_alpha) + self.epsilon

    @property
    def scale(self) -> Tensor:
        return F.softplus(self.raw_scale) + self.epsilon

    @property
    def beta(self) -> Tensor:
        return F.softplus(self.raw_beta) + self.epsilon

    def _edge_parameter(self, parameter: Tensor, edge_type: Tensor) -> Tensor:
        if self.relation_specific:
            return parameter[edge_type]
        return parameter.expand_as(edge_type)

    def forward(
        self,
        source_mass: Tensor,
        destination_mass: Tensor,
        edge_impedance: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        if torch.any(source_mass <= 0) or torch.any(destination_mass <= 0):
            raise ValueError("gravity masses must be strictly positive")
        alpha = self._edge_parameter(self.alpha, edge_type)
        scale = self._edge_parameter(self.scale, edge_type)
        beta = self._edge_parameter(self.beta, edge_type)
        log_weight = -torch.pow(edge_impedance / scale, beta)
        if self.mass_enabled:
            log_mass_product = torch.log(source_mass) + torch.log(destination_mass)
            log_weight = log_weight + alpha * log_mass_product
        return torch.exp(log_weight.clamp(min=-30.0, max=20.0))

    def parameter_summary(self) -> dict[str, list[float]]:
        return {
            "alpha": self.alpha.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
            "beta": self.beta.detach().cpu().tolist(),
        }


class HierarchicalAccessibilityKernel(nn.Module):
    """Monotone source-mass gravity kernel with shrinkage across relations.

    A shared global scale and shape are combined with small relation offsets.
    Bounding the parameters prevents degenerate distance functions in the
    small-area weak-supervision setting. Only source opportunity mass enters
    the accessibility kernel; origin-building mass is a separate feature.
    """

    def __init__(
        self,
        num_relations: int,
        learnable: bool = True,
        initial_scale: float = 15.0,
        initial_beta: float = 1.5,
        scale_bounds: tuple[float, float] = (1.0, 60.0),
        beta_bounds: tuple[float, float] = (0.5, 3.0),
        use_source_mass: bool = False,
        initial_alpha: float = 0.15,
    ) -> None:
        super().__init__()
        if num_relations < 1:
            raise ValueError("num_relations must be positive")
        self.num_relations = num_relations
        self.scale_bounds = scale_bounds
        self.beta_bounds = beta_bounds
        self.use_source_mass = use_source_mass
        self.raw_global_scale = nn.Parameter(
            torch.tensor(
                _inverse_bounded_sigmoid(initial_scale, *scale_bounds),
                dtype=torch.float32,
            ),
            requires_grad=learnable,
        )
        self.raw_global_beta = nn.Parameter(
            torch.tensor(
                _inverse_bounded_sigmoid(initial_beta, *beta_bounds),
                dtype=torch.float32,
            ),
            requires_grad=learnable,
        )
        self.scale_offset = nn.Parameter(
            torch.zeros(num_relations), requires_grad=learnable
        )
        self.beta_offset = nn.Parameter(
            torch.zeros(num_relations), requires_grad=learnable
        )
        self.raw_alpha = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_alpha), dtype=torch.float32),
            requires_grad=learnable and use_source_mass,
        )

    @staticmethod
    def _bounded(raw: Tensor, bounds: tuple[float, float]) -> Tensor:
        lower, upper = bounds
        return lower + (upper - lower) * torch.sigmoid(raw)

    @property
    def scale(self) -> Tensor:
        return self._bounded(
            self.raw_global_scale + self.scale_offset, self.scale_bounds
        )

    @property
    def beta(self) -> Tensor:
        return self._bounded(self.raw_global_beta + self.beta_offset, self.beta_bounds)

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.raw_alpha) + 1e-6

    def forward(
        self, source_mass: Tensor, edge_impedance: Tensor, edge_type: Tensor
    ) -> Tensor:
        scale = self.scale[edge_type]
        beta = self.beta[edge_type]
        log_weight = -torch.pow(edge_impedance / scale, beta)
        if self.use_source_mass:
            if torch.any(source_mass <= 0):
                raise ValueError("source opportunity mass must be positive")
            log_weight = log_weight + self.alpha * torch.log(source_mass)
        return torch.exp(log_weight.clamp(min=-30.0, max=0.0))

    def regularization_loss(self) -> Tensor:
        return self.scale_offset.square().mean() + self.beta_offset.square().mean()

    def parameter_summary(self) -> dict[str, list[float] | float | bool]:
        return {
            "scale": self.scale.detach().cpu().tolist(),
            "beta": self.beta.detach().cpu().tolist(),
            "alpha": float(self.alpha.detach().cpu()),
            "source_mass_enabled": self.use_source_mass,
        }


class GravityAccessibilityEncoder(nn.Module):
    """Create explicit cumulative-opportunity channels for every relation."""

    def __init__(
        self,
        num_accessibility_relations: int,
        learnable: bool,
        use_source_mass: bool = False,
        maximum_impedance: float = 30.0,
    ) -> None:
        super().__init__()
        self.num_relations = num_accessibility_relations
        self.maximum_impedance = maximum_impedance
        self.learnable = learnable
        self._cache_key: tuple[int, int, str] | None = None
        self._cached_structure: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None
        self._cached_fixed_output: Tensor | None = None
        self.kernel = HierarchicalAccessibilityKernel(
            num_accessibility_relations,
            learnable=learnable,
            use_source_mass=use_source_mass,
        )

    @property
    def output_dim(self) -> int:
        return 4 * self.num_relations

    def forward(self, data: GraphData) -> Tensor:
        cache_key = (
            data.edge_index.data_ptr(),
            data.edge_impedance.data_ptr(),
            str(data.x.device),
        )
        if cache_key != self._cache_key:
            source, destination = data.edge_index
            mask = data.edge_type < self.num_relations
            source = source[mask]
            destination = destination[mask]
            relation = data.edge_type[mask]
            impedance = data.edge_impedance[mask]
            flat_index = destination * self.num_relations + relation
            flat_size = data.num_nodes * self.num_relations
            count = data.x.new_zeros(flat_size)
            count.index_add_(0, flat_index, torch.ones_like(impedance))
            minimum = data.x.new_full((flat_size,), float("inf"))
            minimum.scatter_reduce_(
                0, flat_index, impedance, reduce="amin", include_self=True
            )
            missing = count == 0
            minimum = torch.where(
                missing,
                torch.full_like(minimum, self.maximum_impedance),
                minimum.clamp(max=self.maximum_impedance),
            )
            self._cache_key = cache_key
            self._cached_structure = (
                source,
                relation,
                impedance,
                flat_index,
                torch.cat(
                    (
                        torch.log1p(count).reshape(data.num_nodes, self.num_relations),
                        (minimum / self.maximum_impedance).reshape(
                            data.num_nodes, self.num_relations
                        ),
                        missing.to(data.x.dtype).reshape(
                            data.num_nodes, self.num_relations
                        ),
                    ),
                    dim=1,
                ),
            )
            self._cached_fixed_output = None
        if not self.learnable and self._cached_fixed_output is not None:
            return self._cached_fixed_output
        assert self._cached_structure is not None
        source, relation, impedance, flat_index, static_features = self._cached_structure
        weight = self.kernel(data.node_mass[source], impedance, relation)
        flat_size = data.num_nodes * self.num_relations
        potential = data.x.new_zeros(flat_size)
        potential.index_add_(0, flat_index, weight)
        output = torch.cat(
            (
                torch.log1p(potential).reshape(data.num_nodes, self.num_relations),
                static_features,
            ),
            dim=1,
        )
        if not self.learnable:
            self._cached_fixed_output = output.detach()
        return output


class ProximityResidualLayer(nn.Module):
    """Distance-decayed message passing over the building-proximity relation."""

    def __init__(self, hidden_dim: int, proximity_relation: int, dropout: float) -> None:
        super().__init__()
        self.proximity_relation = proximity_relation
        self._cache_key: tuple[int, int, str] | None = None
        self._cached_edges: tuple[Tensor, Tensor, Tensor] | None = None
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, data: GraphData) -> Tensor:
        cache_key = (
            data.edge_index.data_ptr(),
            data.edge_impedance.data_ptr(),
            str(hidden.device),
        )
        if cache_key != self._cache_key:
            source, destination = data.edge_index
            mask = data.edge_type == self.proximity_relation
            source = source[mask]
            destination = destination[mask]
            impedance = data.edge_impedance[mask]
            weight = torch.exp(-torch.pow(impedance / 5.0, 1.5))
            self._cache_key = cache_key
            self._cached_edges = (source, destination, weight)
        assert self._cached_edges is not None
        source, destination, weight = self._cached_edges
        output = hidden.new_zeros(hidden.shape)
        output.index_add_(
            0, destination, self.message(hidden[source]) * weight[:, None]
        )
        degree = hidden.new_zeros(data.num_nodes)
        degree.index_add_(0, destination, torch.ones_like(weight))
        output = output / degree.clamp_min(1.0)[:, None]
        output = self.normalization(self.self_linear(hidden) + output)
        return self.dropout(F.relu(output))


class GravityPotentialRegressor(nn.Module):
    """Explicit gravity-accessibility model with an optional graph residual."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relations: int,
        layers: int,
        dropout: float,
        learnable_kernel: bool,
        graph_residual: bool,
        use_source_mass: bool = False,
    ) -> None:
        super().__init__()
        if num_relations < 2:
            raise ValueError("the v2 model requires accessibility and proximity relations")
        self.accessibility = GravityAccessibilityEncoder(
            num_accessibility_relations=num_relations - 1,
            learnable=learnable_kernel,
            use_source_mass=use_source_mass,
        )
        combined_dim = input_dim + self.accessibility.output_dim
        self.base_encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.base_readout = nn.Linear(hidden_dim, 1)
        self.graph_residual = graph_residual
        if graph_residual:
            self.residual_layers = nn.ModuleList(
                ProximityResidualLayer(hidden_dim, num_relations - 1, dropout)
                for _ in range(max(layers, 1))
            )
            self.residual_readout = nn.Linear(hidden_dim, 1)
            self.raw_residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, data: GraphData) -> Tensor:
        gravity_features = self.accessibility(data)
        hidden = self.base_encoder(torch.cat((data.x, gravity_features), dim=1))
        prediction = self.base_readout(hidden).squeeze(-1)
        if self.graph_residual:
            residual_hidden = hidden
            for layer in self.residual_layers:
                residual_hidden = layer(residual_hidden, data)
            residual = self.residual_readout(residual_hidden).squeeze(-1)
            prediction = prediction + torch.tanh(self.raw_residual_gate) * residual
        return prediction

    def regularization_loss(self) -> Tensor:
        penalty = self.accessibility.kernel.regularization_loss()
        if self.graph_residual:
            penalty = penalty + 0.1 * self.raw_residual_gate.square()
        return penalty

    def kernel_summary(self) -> dict[str, object]:
        summary: dict[str, object] = self.accessibility.kernel.parameter_summary()
        summary["residual_gate"] = (
            float(torch.tanh(self.raw_residual_gate).detach().cpu())
            if self.graph_residual
            else 0.0
        )
        return summary


class GroupPooledGravityRegressor(nn.Module):
    """Predict at the label-group resolution after pooling gravity features.

    The real target is observed once per LSOA. Pooling before the nonlinear
    predictor prevents repeated building rows from acting as fictitious
    independent labels and makes the model output constant within each LSOA.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relations: int,
        layers: int,
        dropout: float,
        learnable_kernel: bool,
        use_source_mass: bool = False,
    ) -> None:
        super().__init__()
        if num_relations < 2:
            raise ValueError("the grouped gravity model requires accessibility and proximity relations")
        self.accessibility = GravityAccessibilityEncoder(
            num_accessibility_relations=num_relations - 1,
            learnable=learnable_kernel,
            use_source_mass=use_source_mass,
        )
        combined_dim = input_dim + self.accessibility.output_dim
        hidden_layers: list[nn.Module] = [nn.Linear(combined_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(layers - 1, 0)):
            hidden_layers.extend((nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        hidden_layers.append(nn.Linear(hidden_dim, 1))
        self.predictor = nn.Sequential(*hidden_layers)

    def forward(self, data: GraphData) -> Tensor:
        if data.label_group is None:
            raise ValueError("group-pooled gravity requires label_group")
        combined = torch.cat((data.x, self.accessibility(data)), dim=1)
        building_features = combined[data.building_mask]
        building_groups = data.label_group[data.building_mask]
        _, inverse = torch.unique(building_groups, sorted=True, return_inverse=True)
        group_count = torch.bincount(inverse).to(dtype=combined.dtype)
        pooled = combined.new_zeros((len(group_count), combined.shape[1]))
        pooled.index_add_(0, inverse, building_features)
        pooled = pooled / group_count[:, None].clamp_min(1.0)
        group_prediction = self.predictor(pooled).squeeze(-1)
        prediction = combined.new_zeros(data.num_nodes)
        prediction[data.building_mask] = group_prediction[inverse]
        return prediction

    def regularization_loss(self) -> Tensor:
        return self.accessibility.kernel.regularization_loss()

    def kernel_summary(self) -> dict[str, object]:
        summary: dict[str, object] = self.accessibility.kernel.parameter_summary()
        summary["group_pooling"] = True
        summary["residual_gate"] = 0.0
        return summary


class FacilityGravityMessageEncoder(nn.Module):
    """Relation-preserving facility-to-area vector message passing.

    Facility nodes in the empirical graph deliberately have no padded tabular
    covariates. Their state is therefore constructed from the edge relation,
    source opportunity mass, and routed travel time. A monotone gravity kernel
    acts as an interpretable prior, while a relation-specific edge network can
    retain the distribution of facility quality and travel time instead of
    reducing every relation to a single scalar potential.

    The output has shape ``[N, R, H]`` so facility relations are not combined
    until the downstream fusion layer.
    """

    def __init__(
        self,
        num_facility_relations: int,
        message_dim: int,
        dropout: float,
        kernel_mode: Literal["learned", "fixed", "uniform", "gated"] = "learned",
        edge_basis: Literal["full", "mass_only"] = "full",
        maximum_impedance: float = 40.0,
        use_source_mass_in_kernel: bool = True,
    ) -> None:
        super().__init__()
        if num_facility_relations < 1:
            raise ValueError("num_facility_relations must be positive")
        if message_dim < 1 or maximum_impedance <= 0:
            raise ValueError("message_dim and maximum_impedance must be positive")
        self.num_relations = num_facility_relations
        self.message_dim = message_dim
        self.maximum_impedance = maximum_impedance
        self.kernel_mode = kernel_mode
        self.edge_basis = edge_basis
        if edge_basis not in {"full", "mass_only"}:
            raise ValueError("edge_basis must be full or mass_only")
        self.kernel = (
            None
            if kernel_mode == "uniform"
            else HierarchicalAccessibilityKernel(
                num_relations=num_facility_relations,
                learnable=kernel_mode in {"learned", "gated"},
                initial_scale=15.0,
                initial_beta=1.5,
                use_source_mass=use_source_mass_in_kernel,
            )
        )
        self.raw_gravity_gate = (
            nn.Parameter(torch.zeros(num_facility_relations))
            if kernel_mode == "gated"
            else None
        )

        # The nonlinear edge bases distinguish quality-distance mixtures even
        # when their scalar gravity sums happen to be equal.
        basis_dim = 6 if edge_basis == "full" else 2
        self.edge_weight = nn.Parameter(
            torch.empty(num_facility_relations, basis_dim, message_dim)
        )
        self.edge_bias = nn.Parameter(
            torch.zeros(num_facility_relations, message_dim)
        )
        self.strength_direction = nn.Parameter(
            torch.empty(num_facility_relations, message_dim)
        )
        self.count_direction = nn.Parameter(
            torch.empty(num_facility_relations, message_dim)
        )
        self.missing_embedding = nn.Parameter(
            torch.zeros(num_facility_relations, message_dim)
        )
        nn.init.xavier_uniform_(self.edge_weight)
        nn.init.xavier_uniform_(self.strength_direction)
        nn.init.xavier_uniform_(self.count_direction)
        self.normalization = nn.LayerNorm(message_dim)
        self.dropout = nn.Dropout(dropout)

    def _edge_basis(self, source_mass: Tensor, impedance: Tensor) -> Tensor:
        log_mass = torch.log(source_mass.clamp_min(1e-6))
        if self.edge_basis == "mass_only":
            return torch.stack((torch.ones_like(log_mass), log_mass), dim=1)
        scaled_time = (impedance / self.maximum_impedance).clamp(min=0.0)
        return torch.stack(
            (
                torch.ones_like(scaled_time),
                log_mass,
                scaled_time,
                scaled_time.square(),
                torch.exp(-2.0 * scaled_time),
                log_mass * scaled_time,
            ),
            dim=1,
        )

    def forward(self, data: GraphData) -> Tensor:
        source, destination = data.edge_index
        output = data.x.new_zeros(
            (data.num_nodes, self.num_relations, self.message_dim)
        )
        for relation in range(self.num_relations):
            mask = data.edge_type == relation
            if not torch.any(mask):
                output[:, relation] = self.missing_embedding[relation]
                continue
            relation_source = source[mask]
            relation_destination = destination[mask]
            impedance = data.edge_impedance[mask]
            source_mass = data.node_mass[relation_source]
            basis = self._edge_basis(source_mass, impedance)
            edge_state = F.gelu(
                basis @ self.edge_weight[relation] + self.edge_bias[relation]
            )
            if self.kernel is None:
                gravity_weight = torch.ones_like(impedance)
            else:
                relation_type = torch.full_like(relation_source, relation)
                gravity_weight = self.kernel(
                    source_mass, impedance, relation_type
                )
                if self.raw_gravity_gate is not None:
                    gravity_gate = torch.sigmoid(self.raw_gravity_gate[relation])
                    gravity_weight = 1.0 + gravity_gate * (gravity_weight - 1.0)

            weighted_sum = data.x.new_zeros((data.num_nodes, self.message_dim))
            weighted_sum.index_add_(
                0,
                relation_destination,
                edge_state * gravity_weight[:, None],
            )
            strength = data.x.new_zeros(data.num_nodes)
            strength.index_add_(0, relation_destination, gravity_weight)
            count = data.x.new_zeros(data.num_nodes)
            count.index_add_(
                0, relation_destination, torch.ones_like(gravity_weight)
            )

            # Weighted mean preserves composition; the two learned directions
            # separately retain total accessibility and opportunity count.
            relation_state = weighted_sum / strength.clamp_min(1e-8)[:, None]
            relation_state = (
                relation_state
                + torch.log1p(strength)[:, None]
                * self.strength_direction[relation]
                + torch.log1p(count)[:, None] * self.count_direction[relation]
            )
            relation_state = relation_state + (count == 0)[:, None] * (
                self.missing_embedding[relation]
            )
            output[:, relation] = relation_state
        return self.dropout(self.normalization(output))

    def regularization_loss(self) -> Tensor:
        if self.kernel is None:
            return self.edge_weight.new_zeros(())
        return self.kernel.regularization_loss()

    def kernel_summary(self) -> dict[str, object]:
        if self.kernel is None:
            return {"mode": "uniform"}
        summary: dict[str, object] = {
            "mode": self.kernel_mode,
            "edge_basis": self.edge_basis,
            **self.kernel.parameter_summary(),
        }
        if self.raw_gravity_gate is not None:
            summary["gravity_gate"] = (
                torch.sigmoid(self.raw_gravity_gate).detach().cpu().tolist()
            )
        return summary


class SplineAreaEncoder(nn.Module):
    """Compact nonlinear tabular encoder using fixed standardized hinges.

    Tree ensembles are strong on the LSOA covariates partly because they learn
    nonlinear thresholds. This encoder gives the neural area branch comparable
    one-dimensional threshold flexibility before it learns cross-feature
    interactions, without using target information to choose the knots.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer(
            "knots", torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        )
        basis_dim = 1 + len(self.knots)
        self.feature_weight = nn.Parameter(
            torch.empty(input_dim, basis_dim, hidden_dim)
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.feature_weight)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        hinges = F.relu(x[:, :, None] - self.knots[None, None, :])
        basis = torch.cat((x[:, :, None], hinges), dim=2)
        additive = torch.einsum("nfk,fkh->nh", basis, self.feature_weight)
        additive = additive / math.sqrt(self.input_dim) + self.bias
        additive = F.gelu(self.normalization(additive))
        output = self.normalization(additive + self.interaction(additive))
        return self.dropout(F.gelu(output))


class HeterogeneousFacilityGravityGNN(nn.Module):
    """Facility-to-area heterogeneous GNN with a gravity edge prior.

    The model creates vector messages on facility edges, fuses the twelve
    relation channels with destination-area covariates, and only then passes
    area states across the final Queen-neighbour relation. Facility nodes and
    individual edges remain in the end-to-end forward computation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relations: int,
        layers: int = 2,
        dropout: float = 0.1,
        kernel_mode: Literal["learned", "fixed", "uniform", "gated"] = "learned",
        edge_basis: Literal["full", "mass_only"] = "full",
        message_dim: int | None = None,
        queen_message_passing: bool = True,
        area_backbone: Literal["mlp", "spline"] = "mlp",
    ) -> None:
        super().__init__()
        if num_relations < 2:
            raise ValueError("facility GNN requires facility and area relations")
        self.num_facility_relations = num_relations - 1
        self.queen_relation = num_relations - 1
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim or max(8, hidden_dim // 4)
        self.facility_messages = FacilityGravityMessageEncoder(
            num_facility_relations=self.num_facility_relations,
            message_dim=self.message_dim,
            dropout=dropout,
            kernel_mode=kernel_mode,
            edge_basis=edge_basis,
            use_source_mass_in_kernel=True,
        )
        self.area_backbone = area_backbone
        self.area_encoder = (
            SplineAreaEncoder(input_dim, hidden_dim, dropout)
            if area_backbone == "spline"
            else nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        )
        relation_dim = self.num_facility_relations * self.message_dim
        self.facility_fusion = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.joint_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.queen_message_passing = queen_message_passing
        self.area_layers = (
            nn.ModuleList(
                ProximityResidualLayer(
                    hidden_dim,
                    proximity_relation=self.queen_relation,
                    dropout=dropout,
                )
                for _ in range(max(layers, 1))
            )
            if queen_message_passing
            else nn.ModuleList()
        )
        self.skip_gate = nn.Parameter(torch.tensor(0.0))
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: GraphData) -> Tensor:
        area_state = data.x.new_zeros((data.num_nodes, self.hidden_dim))
        area_state[data.building_mask] = self.area_encoder(data.x[data.building_mask])
        relation_state = self.facility_messages(data).flatten(start_dim=1)
        facility_state = area_state.new_zeros(area_state.shape)
        facility_state[data.building_mask] = self.facility_fusion(
            relation_state[data.building_mask]
        )
        fused = area_state.new_zeros(area_state.shape)
        fused[data.building_mask] = self.joint_fusion(
            torch.cat(
                (
                    area_state[data.building_mask],
                    facility_state[data.building_mask],
                ),
                dim=1,
            )
        )
        graph_state = fused
        for layer in self.area_layers:
            graph_state = layer(graph_state, data)
        gate = torch.sigmoid(self.skip_gate)
        contextual_state = gate * graph_state + (1.0 - gate) * fused
        prediction = data.x.new_zeros(data.num_nodes)
        prediction_state = torch.cat(
            (
                area_state[data.building_mask],
                contextual_state[data.building_mask],
            ),
            dim=1,
        )
        prediction[data.building_mask] = self.readout(prediction_state).squeeze(-1)
        return prediction

    def regularization_loss(self) -> Tensor:
        return self.facility_messages.regularization_loss()

    def kernel_summary(self) -> dict[str, object]:
        return {
            **self.facility_messages.kernel_summary(),
            "queen_message_passing": self.queen_message_passing,
            "queen_gate": float(torch.sigmoid(self.skip_gate).detach().cpu()),
            "message_dim": self.message_dim,
            "area_backbone": self.area_backbone,
        }


class GravityMessageLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relations: int,
        kernel: GravityKernel | None,
        aggregate: Literal["degree_mean", "weighted_mean", "sum"] = "degree_mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if aggregate not in {"degree_mean", "weighted_mean", "sum"}:
            raise ValueError("aggregate must be degree_mean, weighted_mean, or sum")
        self.num_relations = num_relations
        self.kernel = kernel
        self.aggregate = aggregate
        self.relation_weight = nn.Parameter(
            torch.empty(num_relations, input_dim, output_dim)
        )
        nn.init.xavier_uniform_(self.relation_weight)
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, data: GraphData) -> Tensor:
        source, destination = data.edge_index
        if self.kernel is None:
            edge_weight = torch.ones_like(data.edge_impedance)
        else:
            edge_weight = self.kernel(
                data.node_mass[source],
                data.node_mass[destination],
                data.edge_impedance,
                data.edge_type,
            )

        aggregated = x.new_zeros((data.num_nodes, self.relation_weight.shape[-1]))
        for relation in range(self.num_relations):
            relation_mask = data.edge_type == relation
            if not torch.any(relation_mask):
                continue
            relation_source = source[relation_mask]
            relation_destination = destination[relation_mask]
            relation_messages = x[relation_source] @ self.relation_weight[relation]
            relation_weights = edge_weight[relation_mask]
            relation_output = x.new_zeros(aggregated.shape)
            relation_output.index_add_(
                0,
                relation_destination,
                relation_messages * relation_weights[:, None],
            )
            if self.aggregate in {"degree_mean", "weighted_mean"}:
                denominator = x.new_zeros(data.num_nodes)
                denominator_values = (
                    relation_weights
                    if self.aggregate == "weighted_mean"
                    else torch.ones_like(relation_weights)
                )
                denominator.index_add_(0, relation_destination, denominator_values)
                relation_output = relation_output / denominator.clamp_min(1e-8)[:, None]
            aggregated = aggregated + relation_output

        output = self.self_linear(x) + aggregated
        output = self.normalization(output)
        return self.dropout(F.relu(output))


class GELGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relations: int,
        layers: int = 2,
        dropout: float = 0.1,
        kernel_mode: Literal["learned", "fixed", "uniform"] = "learned",
        relation_specific: bool = True,
        aggregate: Literal["degree_mean", "weighted_mean", "sum"] = "degree_mean",
        mass_enabled: bool = True,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        if kernel_mode == "uniform":
            self.kernel = None
        else:
            self.kernel = GravityKernel(
                num_relations=num_relations,
                relation_specific=relation_specific,
                learnable=kernel_mode == "learned",
                mass_enabled=mass_enabled,
            )
        dimensions = [input_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(
            GravityMessageLayer(
                dimensions[index],
                dimensions[index + 1],
                num_relations=num_relations,
                kernel=self.kernel,
                aggregate=aggregate,
                dropout=dropout,
            )
            for index in range(layers)
        )
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, data: GraphData) -> Tensor:
        x = data.x
        for layer in self.layers:
            x = layer(x, data)
        return self.readout(x).squeeze(-1)

    def kernel_summary(self) -> dict[str, list[float]] | None:
        return None if self.kernel is None else self.kernel.parameter_summary()


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(max(layers, 1)):
            modules.extend(
                [nn.Linear(current, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            current = hidden_dim
        modules.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, data: GraphData) -> Tensor:
        return self.network(data.x).squeeze(-1)


class EdgeAttentionLayer(nn.Module):
    """Single-head relation-aware GAT layer without distance inputs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relations: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.relation_weight = nn.Parameter(
            torch.empty(num_relations, input_dim, output_dim)
        )
        self.attention_source = nn.Parameter(torch.empty(num_relations, output_dim))
        self.attention_destination = nn.Parameter(torch.empty(num_relations, output_dim))
        nn.init.xavier_uniform_(self.relation_weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_destination)
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, data: GraphData) -> Tensor:
        source, destination = data.edge_index
        aggregated = x.new_zeros((data.num_nodes, self.relation_weight.shape[-1]))
        for relation in range(self.num_relations):
            mask = data.edge_type == relation
            if not torch.any(mask):
                continue
            relation_source = source[mask]
            relation_destination = destination[mask]
            projected_source = x[relation_source] @ self.relation_weight[relation]
            projected_destination = x[relation_destination] @ self.relation_weight[relation]
            score = F.leaky_relu(
                (projected_source * self.attention_source[relation]).sum(dim=1)
                + (projected_destination * self.attention_destination[relation]).sum(dim=1),
                negative_slope=0.2,
            )
            maximum = x.new_full((data.num_nodes,), -torch.inf)
            maximum.scatter_reduce_(
                0, relation_destination, score, reduce="amax", include_self=True
            )
            unnormalized = torch.exp((score - maximum[relation_destination]).clamp(max=20.0))
            denominator = x.new_zeros(data.num_nodes)
            denominator.index_add_(0, relation_destination, unnormalized)
            attention = unnormalized / denominator[relation_destination].clamp_min(1e-8)
            relation_output = x.new_zeros(aggregated.shape)
            relation_output.index_add_(
                0, relation_destination, projected_source * attention[:, None]
            )
            aggregated = aggregated + relation_output
        output = self.normalization(self.self_linear(x) + aggregated)
        return self.dropout(F.elu(output))


class GATRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relations: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(
            EdgeAttentionLayer(
                dimensions[index],
                dimensions[index + 1],
                num_relations,
                dropout,
            )
            for index in range(layers)
        )
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, data: GraphData) -> Tensor:
        x = data.x
        for layer in self.layers:
            x = layer(x, data)
        return self.readout(x).squeeze(-1)


def build_model(
    name: str,
    input_dim: int,
    hidden_dim: int,
    num_relations: int,
    layers: int,
    dropout: float,
) -> nn.Module:
    common = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        layers=layers,
        dropout=dropout,
    )
    if name == "mlp":
        return MLPRegressor(input_dim, hidden_dim, layers, dropout)
    if name == "gcn":
        return GELGNN(**common, kernel_mode="uniform", mass_enabled=False)
    if name == "gat":
        return GATRegressor(**common)
    if name == "fixed_gravity":
        return GELGNN(**common, kernel_mode="fixed", relation_specific=True)
    if name == "gel_gnn":
        return GELGNN(**common, kernel_mode="learned", relation_specific=True)
    if name == "gravity_mlp_fixed":
        return GravityPotentialRegressor(
            **common, learnable_kernel=False, graph_residual=False
        )
    if name == "gravity_mlp_learned":
        return GravityPotentialRegressor(
            **common, learnable_kernel=True, graph_residual=False
        )
    if name == "gravity_residual_fixed":
        return GravityPotentialRegressor(
            **common, learnable_kernel=False, graph_residual=True
        )
    if name == "gravity_residual_learned":
        return GravityPotentialRegressor(
            **common, learnable_kernel=True, graph_residual=True
        )
    if name == "gravity_group_fixed":
        return GroupPooledGravityRegressor(
            **common, learnable_kernel=False
        )
    if name == "gravity_group_learned":
        return GroupPooledGravityRegressor(
            **common, learnable_kernel=True
        )
    if name == "hetero_facility_gravity":
        return HeterogeneousFacilityGravityGNN(
            **common, kernel_mode="learned"
        )
    if name == "hetero_facility_no_gravity":
        return HeterogeneousFacilityGravityGNN(
            **common, kernel_mode="uniform"
        )
    if name == "draft_faithful":
        return GELGNN(
            **common,
            kernel_mode="learned",
            relation_specific=False,
            aggregate="sum",
        )
    raise ValueError(f"unknown model: {name}")
