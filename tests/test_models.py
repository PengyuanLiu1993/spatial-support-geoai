import numpy as np
import torch

from gravity_gnn.data import make_synthetic_graph
from gravity_gnn.baselines import multiscale_gravity_features
from gravity_gnn.models import (
    FacilityGravityMessageEncoder,
    GELGNN,
    GravityAccessibilityEncoder,
    GravityKernel,
    GravityPotentialRegressor,
    GroupPooledGravityRegressor,
    HeterogeneousFacilityGravityGNN,
    HierarchicalAccessibilityKernel,
    build_model,
)
from gravity_gnn.training import seed_everything


def test_gravity_kernel_is_positive_and_monotone_in_distance() -> None:
    kernel = GravityKernel(num_relations=2, relation_specific=True)
    distance = torch.linspace(0.0, 40.0, 101)
    edge_type = torch.zeros(101, dtype=torch.long)
    mass = torch.ones(101)
    weight = kernel(mass, mass, distance, edge_type)
    assert torch.all(weight > 0)
    assert torch.all(weight[1:] <= weight[:-1])
    assert kernel.alpha.min() > 0
    assert kernel.scale.min() > 0
    assert kernel.beta.min() > 0


def test_all_models_return_finite_node_predictions() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=3)
    for name in (
        "mlp",
        "gcn",
        "gat",
        "fixed_gravity",
        "gel_gnn",
        "gravity_mlp_fixed",
        "gravity_mlp_learned",
        "gravity_residual_fixed",
        "gravity_residual_learned",
        "gravity_group_fixed",
        "gravity_group_learned",
        "hetero_facility_gravity",
        "hetero_facility_no_gravity",
        "draft_faithful",
    ):
        model = build_model(
            name,
            input_dim=data.num_features,
            hidden_dim=16,
            num_relations=data.num_relations,
            layers=2,
            dropout=0.0,
        )
        prediction = model(data)
        assert prediction.shape == (data.num_nodes,)
        assert torch.isfinite(prediction).all()


def test_learned_kernel_receives_gradients() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=4)
    model = GELGNN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=2,
        dropout=0.0,
    )
    prediction = model(data)
    loss = torch.mean((prediction[data.building_mask] - data.y[data.building_mask]).square())
    loss.backward()
    assert model.kernel is not None
    for parameter in (model.kernel.raw_alpha, model.kernel.raw_scale, model.kernel.raw_beta):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_model_initialisation_is_reproducible_when_seeded_first() -> None:
    seed_everything(123)
    first = build_model("gel_gnn", 10, 8, 2, 2, 0.0)
    seed_everything(123)
    second = build_model("gel_gnn", 10, 8, 2, 2, 0.0)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_default_model_prediction_responds_to_edge_impedance() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=9)
    seed_everything(9)
    model = GELGNN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=2,
        dropout=0.0,
    ).eval()
    near = model(data)
    far_data = type(data)(
        x=data.x,
        node_mass=data.node_mass,
        edge_index=data.edge_index,
        edge_impedance=data.edge_impedance + 20.0,
        edge_type=data.edge_type,
        y=data.y,
        building_mask=data.building_mask,
        group=data.group,
        label_group=data.label_group,
        sample_weight=data.sample_weight,
    )
    far = model(far_data)
    assert not torch.allclose(near[data.building_mask], far[data.building_mask])


def test_hierarchical_accessibility_kernel_is_bounded_and_monotone() -> None:
    kernel = HierarchicalAccessibilityKernel(num_relations=3, learnable=True)
    distance = torch.linspace(0.0, 80.0, 101)
    relation = torch.zeros(101, dtype=torch.long)
    weight = kernel(torch.ones(101), distance, relation)
    assert torch.all(weight[1:] <= weight[:-1])
    assert torch.all((kernel.scale >= 1.0) & (kernel.scale <= 60.0))
    assert torch.all((kernel.beta >= 0.5) & (kernel.beta <= 3.0))


def test_explicit_gravity_potential_increases_with_an_extra_opportunity() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=15)
    encoder = GravityAccessibilityEncoder(
        num_accessibility_relations=1, learnable=False
    )
    first_accessibility_edge = int(torch.nonzero(data.edge_type == 0)[0])
    destination = int(data.edge_index[1, first_accessibility_edge])
    original = encoder(data)
    augmented = type(data)(
        x=data.x,
        node_mass=data.node_mass,
        edge_index=torch.cat(
            (data.edge_index, data.edge_index[:, first_accessibility_edge : first_accessibility_edge + 1]),
            dim=1,
        ),
        edge_impedance=torch.cat(
            (data.edge_impedance, data.edge_impedance[first_accessibility_edge : first_accessibility_edge + 1])
        ),
        edge_type=torch.cat(
            (data.edge_type, data.edge_type[first_accessibility_edge : first_accessibility_edge + 1])
        ),
        y=data.y,
        building_mask=data.building_mask,
        group=data.group,
        label_group=data.label_group,
        sample_weight=data.sample_weight,
    )
    with_extra = encoder(augmented)
    assert with_extra[destination, 0] > original[destination, 0]


def test_v2_learned_kernel_and_residual_gate_receive_gradients() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=21)
    model = GravityPotentialRegressor(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=1,
        dropout=0.0,
        learnable_kernel=True,
        graph_residual=True,
    )
    prediction = model(data)
    loss = torch.mean((prediction[data.building_mask] - data.y[data.building_mask]).square())
    loss.backward()
    kernel = model.accessibility.kernel
    for parameter in (
        kernel.raw_global_scale,
        kernel.raw_global_beta,
        kernel.scale_offset,
        kernel.beta_offset,
        model.raw_residual_gate,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_group_pooled_gravity_is_constant_within_label_group() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=27)
    model = GroupPooledGravityRegressor(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=2,
        dropout=0.0,
        learnable_kernel=True,
    )
    prediction = model(data)
    for group in torch.unique(data.label_group[data.building_mask]):
        selected = data.building_mask & (data.label_group == group)
        assert torch.allclose(prediction[selected], prediction[selected][0].expand_as(prediction[selected]))


def test_multiscale_gravity_features_are_finite() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=31)
    features = multiscale_gravity_features(data)
    assert features.shape == (data.num_nodes, 3)
    assert np.isfinite(features).all()


def test_facility_message_encoder_uses_zero_feature_facility_nodes() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=35)
    facility_sources = torch.unique(data.edge_index[0, data.edge_type == 0])
    data.x[facility_sources] = 0.0
    encoder = FacilityGravityMessageEncoder(
        num_facility_relations=1,
        message_dim=12,
        dropout=0.0,
    ).eval()
    original = encoder(data)
    changed_mass = data.node_mass.clone()
    changed_mass[facility_sources] *= 2.0
    changed_data = type(data)(
        x=data.x,
        node_mass=changed_mass,
        edge_index=data.edge_index,
        edge_impedance=data.edge_impedance,
        edge_type=data.edge_type,
        y=data.y,
        building_mask=data.building_mask,
        group=data.group,
        label_group=data.label_group,
        sample_weight=data.sample_weight,
        feature_names=data.feature_names,
    )
    changed = encoder(changed_data)
    assert torch.isfinite(original).all()
    assert not torch.allclose(
        original[data.building_mask], changed[data.building_mask]
    )


def test_heterogeneous_facility_gnn_responds_to_facility_travel_time() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=37)
    facility_sources = torch.unique(data.edge_index[0, data.edge_type == 0])
    data.x[facility_sources] = 0.0
    seed_everything(37)
    model = HeterogeneousFacilityGravityGNN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=1,
        dropout=0.0,
        queen_message_passing=False,
    ).eval()
    near = model(data)
    far_impedance = data.edge_impedance.clone()
    far_impedance[data.edge_type == 0] += 15.0
    far_data = type(data)(
        x=data.x,
        node_mass=data.node_mass,
        edge_index=data.edge_index,
        edge_impedance=far_impedance,
        edge_type=data.edge_type,
        y=data.y,
        building_mask=data.building_mask,
        group=data.group,
        label_group=data.label_group,
        sample_weight=data.sample_weight,
        feature_names=data.feature_names,
    )
    far = model(far_data)
    assert not torch.allclose(near[data.building_mask], far[data.building_mask])


def test_heterogeneous_facility_kernel_and_messages_receive_gradients() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=41)
    data.x[~data.building_mask] = 0.0
    model = HeterogeneousFacilityGravityGNN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=1,
        dropout=0.0,
    )
    prediction = model(data)
    loss = torch.mean(
        (prediction[data.building_mask] - data.y[data.building_mask]).square()
    )
    loss.backward()
    kernel = model.facility_messages.kernel
    assert kernel is not None
    for parameter in (
        model.facility_messages.edge_weight,
        kernel.raw_global_scale,
        kernel.raw_global_beta,
        kernel.scale_offset,
        kernel.beta_offset,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_gated_gravity_prior_receives_gradients() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=43)
    model = HeterogeneousFacilityGravityGNN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=1,
        dropout=0.0,
        kernel_mode="gated",
    )
    prediction = model(data)
    prediction[data.building_mask].square().mean().backward()
    gate = model.facility_messages.raw_gravity_gate
    assert gate is not None
    assert gate.grad is not None
    assert torch.isfinite(gate.grad).all()


def test_mass_only_edge_basis_makes_kernel_the_only_distance_channel() -> None:
    data = make_synthetic_graph(num_buildings=72, num_amenities=16, seed=47)
    far_impedance = data.edge_impedance.clone()
    far_impedance[data.edge_type == 0] += 15.0
    far_data = type(data)(
        x=data.x,
        node_mass=data.node_mass,
        edge_index=data.edge_index,
        edge_impedance=far_impedance,
        edge_type=data.edge_type,
        y=data.y,
        building_mask=data.building_mask,
        group=data.group,
        label_group=data.label_group,
        sample_weight=data.sample_weight,
        feature_names=data.feature_names,
    )
    seed_everything(47)
    gravity_model = HeterogeneousFacilityGravityGNN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=1,
        dropout=0.0,
        kernel_mode="learned",
        edge_basis="mass_only",
        queen_message_passing=False,
    ).eval()
    seed_everything(47)
    uniform_model = HeterogeneousFacilityGravityGNN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_relations=data.num_relations,
        layers=1,
        dropout=0.0,
        kernel_mode="uniform",
        edge_basis="mass_only",
        queen_message_passing=False,
    ).eval()
    assert not torch.allclose(
        gravity_model(data)[data.building_mask],
        gravity_model(far_data)[data.building_mask],
    )
    assert torch.allclose(
        uniform_model(data)[data.building_mask],
        uniform_model(far_data)[data.building_mask],
    )
