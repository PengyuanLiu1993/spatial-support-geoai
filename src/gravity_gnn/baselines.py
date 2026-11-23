from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

# Avoid joblib's unreliable physical-core probe on non-English Windows hosts.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge, Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import GraphData, GroupSplit
from .training import grouped_regression_metrics, regression_metrics


@dataclass
class BaselineResult:
    training_seconds: float
    metrics: dict[str, dict[str, float]]


def _context_feature_mask(
    feature_names: tuple[str, ...], include_spatial: bool = True
) -> np.ndarray:
    names = np.asarray(feature_names)
    selected = (
        np.char.startswith(names, "spatial_")
        | np.char.startswith(names, "log_full_")
        | np.char.startswith(names, "log_primary_")
        | np.char.startswith(names, "log_hospital_")
        | np.char.startswith(names, "log_physical_")
        | np.char.startswith(names, "log_fast_")
        | np.char.startswith(names, "log_pub_")
        | np.char.startswith(names, "log_gambling_")
        | np.char.startswith(names, "log_tobacco_")
        | np.char.startswith(names, "log_food_")
        | np.char.startswith(names, "log_education_")
        | np.char.startswith(names, "log_social_")
        | np.char.startswith(names, "log_major_")
        | np.char.startswith(names, "log_local_")
        | np.char.startswith(names, "log_active_")
        | np.char.startswith(names, "log_os_green_")
        | np.char.startswith(names, "log_nhs_")
        | np.char.startswith(names, "log_fsa_")
        | np.char.startswith(names, "log_gambling_register_")
        | np.char.startswith(names, "defra_")
    )
    if not include_spatial:
        selected &= ~np.char.startswith(names, "spatial_")
    return selected


def edge_summary_features(data: GraphData) -> np.ndarray:
    """Return per-node minimum/mean impedance and edge count by relation."""

    destination = data.edge_index[1].detach().cpu().numpy()
    relation = data.edge_type.detach().cpu().numpy()
    impedance = data.edge_impedance.detach().cpu().numpy()
    flat_index = destination * data.num_relations + relation
    size = data.num_nodes * data.num_relations
    minimum = np.full(size, np.inf, dtype=np.float32)
    total = np.zeros(size, dtype=np.float32)
    count = np.zeros(size, dtype=np.float32)
    np.minimum.at(minimum, flat_index, impedance)
    np.add.at(total, flat_index, impedance)
    np.add.at(count, flat_index, 1.0)
    minimum[count == 0] = np.nan
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return np.concatenate(
        (
            minimum.reshape(data.num_nodes, data.num_relations),
            mean.reshape(data.num_nodes, data.num_relations),
            np.log1p(count.reshape(data.num_nodes, data.num_relations)),
        ),
        axis=1,
    )


def multiscale_gravity_features(
    data: GraphData, scales: tuple[float, ...] = (5.0, 15.0, 30.0)
) -> np.ndarray:
    """Return stable cumulative-opportunity bases at several travel-time scales."""

    destination = data.edge_index[1].detach().cpu().numpy()
    relation = data.edge_type.detach().cpu().numpy()
    impedance = data.edge_impedance.detach().cpu().numpy()
    source = data.edge_index[0].detach().cpu().numpy()
    accessibility_relations = max(data.num_relations - 1, 1)
    selected = relation < accessibility_relations
    destination = destination[selected]
    relation = relation[selected]
    impedance = impedance[selected]
    source_mass = data.node_mass.detach().cpu().numpy()[source[selected]]
    flat_index = destination * accessibility_relations + relation
    flat_size = data.num_nodes * accessibility_relations
    blocks: list[np.ndarray] = []
    for scale in scales:
        potential = np.zeros(flat_size, dtype=np.float32)
        weight = np.power(source_mass, 0.15) * np.exp(-np.power(impedance / scale, 1.5))
        np.add.at(potential, flat_index, weight)
        blocks.append(np.log1p(potential.reshape(data.num_nodes, accessibility_relations)))
    return np.concatenate(blocks, axis=1)


def fit_tabular_baseline(
    name: str,
    data: GraphData,
    split: GroupSplit,
    seed: int,
) -> BaselineResult:
    x = data.x.detach().cpu().numpy()
    base_name = name
    context_dim: int | None = None
    if name == "context_pollution_blend":
        if data.feature_names is None:
            raise ValueError(f"{name} requires feature_names")
        names = np.asarray(data.feature_names)
        context_x = x[:, _context_feature_mask(data.feature_names)]
        pollution_x = x[:, np.char.startswith(names, "defra_")]
        context_dim = context_x.shape[1]
        x = np.concatenate((context_x, pollution_x), axis=1)
        base_name = name
    elif name == "context_gravity_residual":
        if data.feature_names is None:
            raise ValueError(f"{name} requires feature_names")
        context_x = x[:, _context_feature_mask(data.feature_names)]
        gravity_x = np.concatenate(
            (edge_summary_features(data), multiscale_gravity_features(data)), axis=1
        )
        context_dim = context_x.shape[1]
        x = np.concatenate((context_x, gravity_x), axis=1)
        base_name = name
    elif name in {
        "spatial_ridge",
        "context_ridge",
        "mechanism_ridge",
        "reconstruction_ridge",
        "pollution_ridge",
        "pollution_ridge_cv",
        "context_pls",
        "context_bayesian_ridge",
        "context_ridge_spatial",
    }:
        if data.feature_names is None:
            raise ValueError(f"{name} requires feature_names")
        names = np.asarray(data.feature_names)
        if name == "spatial_ridge":
            selected = np.char.startswith(names, "spatial_")
        elif name.startswith("pollution_"):
            selected = np.char.startswith(names, "defra_")
        else:
            selected = _context_feature_mask(
                data.feature_names, include_spatial=name != "mechanism_ridge"
            )
        x = x[:, selected]
        if name == "reconstruction_ridge":
            x = np.concatenate(
                (x, edge_summary_features(data), multiscale_gravity_features(data)), axis=1
            )
        if name == "pollution_ridge_cv":
            base_name = "ridge_cv"
        elif name == "context_pls":
            base_name = "pls_spatial"
        elif name == "context_bayesian_ridge":
            base_name = "bayesian_ridge"
        elif name == "context_ridge_spatial":
            base_name = "ridge_spatial"
        else:
            base_name = "ridge"
    elif name.startswith("hybrid_"):
        x = np.concatenate(
            (x, edge_summary_features(data), multiscale_gravity_features(data)), axis=1
        )
        base_name = name.removeprefix("hybrid_")
    elif name.startswith("gravity_"):
        x = np.concatenate((x, multiscale_gravity_features(data)), axis=1)
        base_name = name.removeprefix("gravity_")
    elif name.startswith("edge_"):
        x = np.concatenate((x, edge_summary_features(data)), axis=1)
        base_name = name.removeprefix("edge_")
    y = data.y.detach().cpu().numpy()
    train = split.train.detach().cpu().numpy()
    validation = split.validation.detach().cpu().numpy()
    test = split.test.detach().cpu().numpy()
    sample_weight = (
        data.sample_weight.detach().cpu().numpy() if data.sample_weight is not None else None
    )
    grouped_fit = data.label_group is not None
    if grouped_fit:
        label_group = data.label_group.detach().cpu().numpy()
        labelled = data.building_mask.detach().cpu().numpy()
        group_ids = np.unique(label_group[labelled])
        grouped_x = np.vstack([np.mean(x[label_group == group], axis=0) for group in group_ids])
        grouped_y = np.array([np.mean(y[label_group == group]) for group in group_ids])
        train_groups = set(label_group[train].tolist())
        validation_groups = set(label_group[validation].tolist())
        test_groups = set(label_group[test].tolist())
        train = np.array([group in train_groups for group in group_ids])
        validation = np.array([group in validation_groups for group in group_ids])
        test = np.array([group in test_groups for group in group_ids])
        x = grouped_x
        y = grouped_y
        sample_weight = None
    if base_name == "dummy_mean":
        started = time.perf_counter()
        if sample_weight is None:
            mean = float(np.mean(y[train]))
        else:
            mean = float(np.average(y[train], weights=sample_weight[train]))
        prediction = torch.full((len(y),), mean, dtype=torch.float32)
        elapsed = time.perf_counter() - started
        model = None
        fit_kwargs = {}
    elif base_name in {"ridge", "ridge_cv"}:
        ridge = (
            Ridge(alpha=10.0)
            if base_name == "ridge"
            else RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))
        )
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), ridge
        )
        ridge_step = "ridge" if base_name == "ridge" else "ridgecv"
        fit_kwargs = (
            {f"{ridge_step}__sample_weight": sample_weight[train]}
            if sample_weight is not None
            else {}
        )
    elif base_name == "ridge_spatial":
        started = time.perf_counter()
        best_validation = float("inf")
        best_prediction: np.ndarray | None = None
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
            candidate = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
            )
            candidate_kwargs = (
                {"ridge__sample_weight": sample_weight[train]}
                if sample_weight is not None
                else {}
            )
            candidate.fit(x[train], y[train], **candidate_kwargs)
            candidate_prediction = candidate.predict(x)
            validation_rmse = float(
                np.sqrt(np.mean(np.square(candidate_prediction[validation] - y[validation])))
            )
            if validation_rmse < best_validation:
                best_validation = validation_rmse
                best_prediction = candidate_prediction
        assert best_prediction is not None
        prediction = torch.as_tensor(best_prediction, dtype=torch.float32)
        elapsed = time.perf_counter() - started
        model = None
        fit_kwargs = {}
    elif base_name == "pls_spatial":
        started = time.perf_counter()
        best_validation = float("inf")
        best_prediction: np.ndarray | None = None
        for components in (1, 2, 3, 5, 8, 12, 16):
            if components > min(int(train.sum()) - 1, x.shape[1]):
                continue
            candidate = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                PLSRegression(n_components=components, scale=False, max_iter=1000),
            )
            candidate.fit(x[train], y[train])
            candidate_prediction = np.asarray(candidate.predict(x)).reshape(-1)
            validation_rmse = float(
                np.sqrt(np.mean(np.square(candidate_prediction[validation] - y[validation])))
            )
            if validation_rmse < best_validation:
                best_validation = validation_rmse
                best_prediction = candidate_prediction
        assert best_prediction is not None
        prediction = torch.as_tensor(best_prediction, dtype=torch.float32)
        elapsed = time.perf_counter() - started
        model = None
        fit_kwargs = {}
    elif base_name == "bayesian_ridge":
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), BayesianRidge()
        )
        fit_kwargs = {}
    elif base_name == "context_gravity_residual":
        assert context_dim is not None
        started = time.perf_counter()
        context_model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
        )
        context_model.fit(x[train, :context_dim], y[train])
        context_prediction = context_model.predict(x[:, :context_dim])
        train_residual = y[train] - context_prediction[train]
        best_validation = float("inf")
        best_prediction: np.ndarray | None = None
        for alpha in (10.0, 100.0, 1000.0, 10000.0):
            residual_model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
            )
            residual_model.fit(x[train, context_dim:], train_residual)
            residual_prediction = residual_model.predict(x[:, context_dim:])
            for gate in (0.0, 0.25, 0.5, 0.75, 1.0):
                candidate_prediction = context_prediction + gate * residual_prediction
                validation_rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(candidate_prediction[validation] - y[validation])
                        )
                    )
                )
                if validation_rmse < best_validation:
                    best_validation = validation_rmse
                    best_prediction = candidate_prediction
        assert best_prediction is not None
        prediction = torch.as_tensor(best_prediction, dtype=torch.float32)
        elapsed = time.perf_counter() - started
        model = None
        fit_kwargs = {}
    elif base_name == "context_pollution_blend":
        assert context_dim is not None
        started = time.perf_counter()
        context_model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
        )
        pollution_model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)),
        )
        context_model.fit(x[train, :context_dim], y[train])
        pollution_model.fit(x[train, context_dim:], y[train])
        context_prediction = context_model.predict(x[:, :context_dim])
        pollution_prediction = pollution_model.predict(x[:, context_dim:])
        best_validation = float("inf")
        best_prediction: np.ndarray | None = None
        for pollution_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            candidate_prediction = (
                (1.0 - pollution_weight) * context_prediction
                + pollution_weight * pollution_prediction
            )
            validation_rmse = float(
                np.sqrt(np.mean(np.square(candidate_prediction[validation] - y[validation])))
            )
            if validation_rmse < best_validation:
                best_validation = validation_rmse
                best_prediction = candidate_prediction
        assert best_prediction is not None
        prediction = torch.as_tensor(best_prediction, dtype=torch.float32)
        elapsed = time.perf_counter() - started
        model = None
        fit_kwargs = {}
    elif base_name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=seed,
        )
        fit_kwargs = (
            {"sample_weight": sample_weight[train]} if sample_weight is not None else {}
        )
    else:
        raise ValueError(f"unknown tabular baseline: {name}")

    if model is not None:
        started = time.perf_counter()
        model.fit(x[train], y[train], **fit_kwargs)
        prediction = torch.as_tensor(model.predict(x), dtype=torch.float32)
        elapsed = time.perf_counter() - started
    target = data.y.detach().cpu()
    train_tensor = split.train.detach().cpu()
    validation_tensor = split.validation.detach().cpu()
    test_tensor = split.test.detach().cpu()
    if grouped_fit:
        grouped_prediction = prediction
        prediction = torch.zeros(data.num_nodes, dtype=torch.float32)
        label_group_tensor = data.label_group.detach().cpu()
        for index, group in enumerate(group_ids):
            prediction[label_group_tensor == int(group)] = grouped_prediction[index]
    metrics = {
        "train": regression_metrics(target[train_tensor], prediction[train_tensor]),
        "validation": regression_metrics(
            target[validation_tensor], prediction[validation_tensor]
        ),
        "test": regression_metrics(target[test_tensor], prediction[test_tensor]),
    }
    if data.label_group is not None:
        label_group = data.label_group.detach().cpu()
        metrics.update(
            {
                "train_grouped": grouped_regression_metrics(
                    target, prediction, label_group, train_tensor
                ),
                "validation_grouped": grouped_regression_metrics(
                    target, prediction, label_group, validation_tensor
                ),
                "test_grouped": grouped_regression_metrics(
                    target, prediction, label_group, test_tensor
                ),
            }
        )
    return BaselineResult(training_seconds=elapsed, metrics=metrics)
