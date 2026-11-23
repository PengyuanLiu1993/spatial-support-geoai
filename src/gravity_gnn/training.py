from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .data import GraphData, GroupSplit


@dataclass
class FitResult:
    predictions: Tensor
    best_epoch: int
    training_seconds: float
    metrics: dict[str, dict[str, float]]


@dataclass
class FixedEpochFitResult:
    predictions: Tensor
    training_seconds: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    target = target.detach().float().cpu()
    prediction = prediction.detach().float().cpu()
    residual = prediction - target
    mse = torch.mean(residual.square())
    mae = torch.mean(residual.abs())
    total = torch.sum((target - target.mean()).square())
    r_squared = 1.0 - torch.sum(residual.square()) / total.clamp_min(1e-12)
    smape = torch.mean(
        2.0 * residual.abs() / (target.abs() + prediction.abs()).clamp_min(1e-6)
    )
    return {
        "mae": float(mae),
        "rmse": float(torch.sqrt(mse)),
        "r2": float(r_squared),
        "smape_percent": float(100.0 * smape),
    }


def grouped_regression_metrics(
    target: Tensor,
    prediction: Tensor,
    group: Tensor,
    mask: Tensor,
) -> dict[str, float]:
    selected_groups = group[mask]
    unique_groups, inverse = torch.unique(selected_groups, return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(unique_groups)).float()
    target_sum = torch.zeros(len(unique_groups), device=target.device).index_add_(
        0, inverse, target[mask]
    )
    prediction_sum = torch.zeros(
        len(unique_groups), device=prediction.device
    ).index_add_(0, inverse, prediction[mask])
    return regression_metrics(target_sum / counts, prediction_sum / counts)


def grouped_mse_loss(
    target: Tensor,
    prediction: Tensor,
    group: Tensor,
    mask: Tensor,
) -> Tensor:
    """Mean squared error after aggregating predictions to supervision groups."""

    selected_groups = group[mask]
    unique_groups, inverse = torch.unique(selected_groups, return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(unique_groups)).to(
        dtype=prediction.dtype, device=prediction.device
    )
    target_sum = torch.zeros(
        len(unique_groups), dtype=target.dtype, device=target.device
    ).index_add_(0, inverse, target[mask])
    prediction_sum = torch.zeros(
        len(unique_groups), dtype=prediction.dtype, device=prediction.device
    ).index_add_(0, inverse, prediction[mask])
    target_mean = target_sum / counts
    prediction_mean = prediction_sum / counts
    return torch.mean((prediction_mean - target_mean).square())


def standardize_building_features(
    data: GraphData, train_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit scaling on training buildings and leave context-node encodings intact."""

    train_x = data.x[train_mask]
    if data.sample_weight is None:
        feature_mean = train_x.mean(dim=0)
        feature_variance = (train_x - feature_mean).square().mean(dim=0)
    else:
        weights = data.sample_weight[train_mask].unsqueeze(1)
        weight_sum = weights.sum().clamp_min(1e-8)
        feature_mean = torch.sum(weights * train_x, dim=0) / weight_sum
        feature_variance = (
            torch.sum(weights * (train_x - feature_mean).square(), dim=0) / weight_sum
        )
    feature_scale = torch.sqrt(feature_variance).clamp_min(1e-6)
    standardized = data.x.clone()
    standardized[data.building_mask] = (
        data.x[data.building_mask] - feature_mean
    ) / feature_scale
    return standardized, feature_mean, feature_scale


def _weighted_mean_and_scale(
    values: Tensor, weights: Tensor | None
) -> tuple[Tensor, Tensor]:
    if weights is None:
        mean = values.mean()
        variance = (values - mean).square().mean()
    else:
        weight_sum = weights.sum().clamp_min(1e-8)
        mean = torch.sum(weights * values) / weight_sum
        variance = torch.sum(weights * (values - mean).square()) / weight_sum
    return mean, torch.sqrt(variance).clamp_min(1e-6)


def _prepare_group_mean(
    target: Tensor, group: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    _, inverse = torch.unique(group[mask], return_inverse=True)
    counts = torch.bincount(inverse).to(dtype=target.dtype, device=target.device)
    target_sum = torch.zeros(len(counts), dtype=target.dtype, device=target.device)
    target_sum.index_add_(0, inverse, target[mask])
    return inverse, counts, target_sum / counts


def _prepared_group_mse(
    prediction: Tensor,
    mask: Tensor,
    inverse: Tensor,
    counts: Tensor,
    target_mean: Tensor,
) -> Tensor:
    prediction_sum = torch.zeros(
        len(counts), dtype=prediction.dtype, device=prediction.device
    )
    prediction_sum.index_add_(0, inverse, prediction[mask])
    return torch.mean((prediction_sum / counts - target_mean).square())


def fit_model(
    model: nn.Module,
    data: GraphData,
    split: GroupSplit,
    epochs: int = 120,
    patience: int = 20,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str | torch.device = "cpu",
    normalize_features: bool = True,
) -> FitResult:
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    seed_everything(seed)
    device = torch.device(device)
    data = data.to(device)
    train_mask = split.train.to(device)
    validation_mask = split.validation.to(device)
    test_mask = split.test.to(device)
    model = model.to(device)

    if normalize_features:
        standardized_x, _, _ = standardize_building_features(data, train_mask)
        data = replace(data, x=standardized_x)

    train_weights = (
        data.sample_weight[train_mask] if data.sample_weight is not None else None
    )
    target_mean, target_scale = _weighted_mean_and_scale(
        data.y[train_mask], train_weights
    )
    standardized_target = (data.y - target_mean) / target_scale
    train_grouping = (
        _prepare_group_mean(
            standardized_target, data.label_group, train_mask
        )
        if data.label_group is not None
        else None
    )
    validation_grouping = (
        _prepare_group_mean(
            standardized_target, data.label_group, validation_mask
        )
        if data.label_group is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        standardized_prediction = model(data)
        if data.label_group is not None:
            assert train_grouping is not None
            loss = _prepared_group_mse(
                standardized_prediction, train_mask, *train_grouping
            )
        else:
            squared_error = (
                standardized_prediction[train_mask] - standardized_target[train_mask]
            ).square()
            if data.sample_weight is None:
                loss = torch.mean(squared_error)
            else:
                weights = data.sample_weight[train_mask]
                loss = torch.sum(weights * squared_error) / weights.sum().clamp_min(1e-8)
        regularization = getattr(model, "regularization_loss", None)
        if regularization is not None:
            loss = loss + 1e-3 * regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(data)
            if data.label_group is not None:
                assert validation_grouping is not None
                validation_loss = _prepared_group_mse(
                    validation_prediction, validation_mask, *validation_grouping
                ).item()
            else:
                validation_error = (
                    validation_prediction[validation_mask]
                    - standardized_target[validation_mask]
                ).square()
                if data.sample_weight is None:
                    validation_loss = torch.mean(validation_error).item()
                else:
                    validation_weights = data.sample_weight[validation_mask]
                    validation_loss = (
                        torch.sum(validation_weights * validation_error)
                        / validation_weights.sum().clamp_min(1e-8)
                    ).item()
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(data) * target_scale + target_mean
    elapsed = time.perf_counter() - started

    metrics = {
        "train": regression_metrics(data.y[train_mask], predictions[train_mask]),
        "validation": regression_metrics(
            data.y[validation_mask], predictions[validation_mask]
        ),
        "test": regression_metrics(data.y[test_mask], predictions[test_mask]),
    }
    if data.label_group is not None:
        metrics.update(
            {
                "train_grouped": grouped_regression_metrics(
                    data.y, predictions, data.label_group, train_mask
                ),
                "validation_grouped": grouped_regression_metrics(
                    data.y, predictions, data.label_group, validation_mask
                ),
                "test_grouped": grouped_regression_metrics(
                    data.y, predictions, data.label_group, test_mask
                ),
            }
        )
    return FitResult(
        predictions=predictions.detach().cpu(),
        best_epoch=best_epoch,
        training_seconds=elapsed,
        metrics=metrics,
    )


def fit_model_fixed_epochs(
    model: nn.Module,
    data: GraphData,
    train_mask: Tensor,
    epochs: int,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str | torch.device = "cpu",
    normalize_features: bool = True,
) -> FixedEpochFitResult:
    """Refit a selected model on all outer-training observations.

    Early stopping first selects ``epochs`` on an inner spatial validation fold.
    This function then trains a freshly initialized model for exactly that many
    epochs after returning the validation fold to the outer training set.
    """

    if epochs < 1:
        raise ValueError("epochs must be positive")
    seed_everything(seed)
    device = torch.device(device)
    data = data.to(device)
    train_mask = train_mask.to(device)
    model = model.to(device)

    if normalize_features:
        standardized_x, _, _ = standardize_building_features(data, train_mask)
        data = replace(data, x=standardized_x)

    train_weights = (
        data.sample_weight[train_mask] if data.sample_weight is not None else None
    )
    target_mean, target_scale = _weighted_mean_and_scale(
        data.y[train_mask], train_weights
    )
    standardized_target = (data.y - target_mean) / target_scale
    train_grouping = (
        _prepare_group_mean(standardized_target, data.label_group, train_mask)
        if data.label_group is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        standardized_prediction = model(data)
        if train_grouping is not None:
            loss = _prepared_group_mse(
                standardized_prediction, train_mask, *train_grouping
            )
        else:
            squared_error = (
                standardized_prediction[train_mask]
                - standardized_target[train_mask]
            ).square()
            if train_weights is None:
                loss = torch.mean(squared_error)
            else:
                loss = torch.sum(train_weights * squared_error) / train_weights.sum().clamp_min(
                    1e-8
                )
        regularization = getattr(model, "regularization_loss", None)
        if regularization is not None:
            loss = loss + 1e-3 * regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(data) * target_scale + target_mean
    return FixedEpochFitResult(
        predictions=predictions.detach().cpu(),
        training_seconds=time.perf_counter() - started,
    )
