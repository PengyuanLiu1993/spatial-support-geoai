from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


GRAVITY_COLUMNS = (
    "GP",
    "dentist",
    "pharmacy",
    "hospital",
    "leisure",
    "greenspace_active",
    "fast_food",
    "gambling",
    "pub_bar",
    "tobacco",
)

TRUE_SCALES_MINUTES = {
    "GP": 8.0,
    "dentist": 8.0,
    "pharmacy": 6.0,
    "hospital": 18.0,
    "leisure": 12.0,
    "greenspace_active": 10.0,
    "fast_food": 6.0,
    "gambling": 7.0,
    "pub_bar": 6.0,
    "tobacco": 7.0,
}

# Positive values increase the synthetic unhealthy-environment score.
TRUE_WEIGHTS = {
    "GP": -1.00,
    "dentist": -0.75,
    "pharmacy": -0.85,
    "hospital": -0.55,
    "leisure": -0.65,
    "greenspace_active": -0.80,
    "fast_food": 1.00,
    "gambling": 0.85,
    "pub_bar": 0.65,
    "tobacco": 0.75,
}


def gravity_features(
    travel_minutes: np.ndarray,
    columns: Sequence[str] = GRAVITY_COLUMNS,
    scales: Mapping[str, float] = TRUE_SCALES_MINUTES,
) -> np.ndarray:
    values = np.asarray(travel_minutes, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError("travel_minutes shape does not match gravity columns")
    scale = np.asarray([float(scales[column]) for column in columns], dtype=float)
    if np.any(scale <= 0):
        raise ValueError("gravity scales must be positive")
    return np.exp(-np.maximum(values, 0.0) / scale)


def raw_gravity_signal(
    travel_minutes: np.ndarray,
    coordinates: np.ndarray,
    columns: Sequence[str] = GRAVITY_COLUMNS,
    spatial_strength: float = 0.12,
) -> np.ndarray:
    features = gravity_features(travel_minutes, columns)
    weights = np.asarray([TRUE_WEIGHTS[column] for column in columns], dtype=float)
    signal = features @ weights
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape != (len(signal), 2):
        raise ValueError("coordinates must have shape [observations, 2]")
    spatial = np.sin(xy[:, 0] / 17000.0) + 0.7 * np.cos(xy[:, 1] / 23000.0)
    return signal + spatial_strength * spatial


def multiscale_gravity_features(
    travel_minutes: np.ndarray,
    scales: Sequence[float] = (3.0, 6.0, 10.0, 18.0, 30.0),
) -> np.ndarray:
    values = np.maximum(np.asarray(travel_minutes, dtype=float), 0.0)
    blocks = [np.exp(-values / float(scale)) for scale in scales]
    return np.concatenate(blocks, axis=1)


def coordinate_features(coordinates: np.ndarray) -> np.ndarray:
    xy = np.asarray(coordinates, dtype=float)
    centre = xy.mean(axis=0)
    scale = np.maximum(xy.std(axis=0), 1.0)
    normalised = (xy - centre) / scale
    x = normalised[:, 0]
    y = normalised[:, 1]
    return np.column_stack((x, y, x * x, y * y, x * y))


def relative_spatial_basis(
    coordinates: np.ndarray,
    harmonics: int = 3,
    rbf_centres: Sequence[float] = (-1.5, -0.5, 0.5, 1.5),
    rbf_width: float = 0.8,
) -> np.ndarray:
    """City-relative smooth basis; useful only as an explicitly tested spatial residual."""

    if harmonics < 0:
        raise ValueError("harmonics must be non-negative")
    if rbf_width <= 0:
        raise ValueError("rbf_width must be positive")
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape [observations, 2]")
    centre = xy.mean(axis=0)
    scale = np.maximum(xy.std(axis=0), 1.0)
    normalised = (xy - centre) / scale
    x = normalised[:, 0]
    y = normalised[:, 1]
    blocks = [np.column_stack((x, y, x * x, y * y, x * y))]
    for frequency in range(1, harmonics + 1):
        blocks.append(
            np.column_stack(
                (
                    np.sin(frequency * x),
                    np.cos(frequency * x),
                    np.sin(frequency * y),
                    np.cos(frequency * y),
                    np.sin(frequency * (x + y)),
                    np.cos(frequency * (x + y)),
                    np.sin(frequency * (x - y)),
                    np.cos(frequency * (x - y)),
                )
            )
        )
    centres = np.asarray(tuple(float(value) for value in rbf_centres), dtype=float)
    if len(centres):
        for centre_x in centres:
            for centre_y in centres:
                squared_distance = (x - centre_x) ** 2 + (y - centre_y) ** 2
                blocks.append(np.exp(-0.5 * squared_distance / (rbf_width**2))[:, None])
    return np.column_stack(blocks)
