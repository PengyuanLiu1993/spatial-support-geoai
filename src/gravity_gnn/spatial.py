from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def gravity_neighborhood_features(
    coordinates: np.ndarray,
    features: np.ndarray,
    source_mass: np.ndarray,
    *,
    scales_m: tuple[float, ...] = (1000.0, 3000.0, 5000.0, 10000.0),
    neighbors: int = 256,
) -> np.ndarray:
    """Target-free, distance-decayed neighborhood summaries for area nodes.

    The source mass controls opportunity strength, while an exponential gravity
    kernel controls decay. The focal node is excluded, so the summaries measure
    surrounding context rather than duplicating its own covariates.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    features = np.asarray(features, dtype=float)
    source_mass = np.asarray(source_mass, dtype=float).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [n, 2]")
    if features.ndim != 2 or len(features) != len(coordinates):
        raise ValueError("features must have shape [n, p]")
    if source_mass.shape != (len(coordinates),):
        raise ValueError("source_mass must have one value per node")
    if not scales_m or any(scale <= 0 for scale in scales_m):
        raise ValueError("gravity scales must be positive")
    if len(coordinates) < 2:
        return np.zeros((len(coordinates), len(scales_m) * (1 + 2 * features.shape[1])))

    finite_features = np.where(np.isfinite(features), features, np.nan)
    column_medians = np.nanmedian(finite_features, axis=0)
    column_medians = np.where(np.isfinite(column_medians), column_medians, 0.0)
    finite_features = np.where(np.isfinite(finite_features), finite_features, column_medians)
    mass = np.where(np.isfinite(source_mass) & (source_mass > 0), source_mass, np.nan)
    median_mass = np.nanmedian(mass)
    if not np.isfinite(median_mass) or median_mass <= 0:
        median_mass = 1.0
    mass = np.nan_to_num(mass / median_mass, nan=1.0, posinf=10.0, neginf=0.1)
    mass = np.clip(mass, 0.1, 10.0)

    k = min(max(int(neighbors), 1) + 1, len(coordinates))
    distances, indices = cKDTree(coordinates).query(coordinates, k=k)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    neighbor_features = finite_features[indices]
    neighbor_mass = mass[indices]

    output = []
    for scale in scales_m:
        weights = neighbor_mass * np.exp(-distances / float(scale))
        denominator = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        neighbor_mean = np.einsum("nk,nkp->np", weights, neighbor_features) / denominator
        potential = np.log1p(weights.sum(axis=1, keepdims=True))
        output.extend((potential, neighbor_mean, finite_features - neighbor_mean))
    return np.concatenate(output, axis=1)
