"""Pré-processamento da curva Gamma Ray: ordenação, limpeza e normalização."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NormalizationParams:
    mean: float
    std: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def invert(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def sort_by_depth(depth: np.ndarray, gamma_ray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(depth)
    return depth[order], gamma_ray[order]


def drop_duplicate_depths(depth: np.ndarray, gamma_ray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth_unique, first_idx = np.unique(depth, return_index=True)
    return depth_unique, gamma_ray[first_idx]


def interpolate_missing(depth: np.ndarray, gamma_ray: np.ndarray) -> np.ndarray:
    """Interpola valores ausentes (NaN) e inválidos (ex.: -999.25) linearmente."""
    gr = gamma_ray.copy()
    gr[gr <= -999.0] = np.nan

    valid = ~np.isnan(gr)
    if valid.sum() < 2:
        raise ValueError("Curva com menos de 2 pontos válidos; não é possível interpolar.")

    gr[~valid] = np.interp(depth[~valid], depth[valid], gr[valid])
    return gr


def clip_outliers(gamma_ray: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Limita valores extremos (ex.: erros de ferramenta) aos percentis informados."""
    low, high = np.percentile(gamma_ray, [low_pct, high_pct])
    return np.clip(gamma_ray, low, high)


def fit_normalization(gamma_ray: np.ndarray) -> NormalizationParams:
    return NormalizationParams(mean=float(np.mean(gamma_ray)), std=float(np.std(gamma_ray) + 1e-8))


def preprocess_curve(
    depth: np.ndarray,
    gamma_ray: np.ndarray,
    clip_pct: tuple[float, float] = (0.5, 99.5),
) -> tuple[np.ndarray, np.ndarray, NormalizationParams]:
    """Pipeline completo: ordena, remove duplicatas, interpola, faz clipping e normaliza.

    Retorna (depth_limpo, gamma_ray_normalizado, params_normalizacao).
    """
    depth, gamma_ray = sort_by_depth(depth, gamma_ray)
    depth, gamma_ray = drop_duplicate_depths(depth, gamma_ray)
    gamma_ray = interpolate_missing(depth, gamma_ray)
    gamma_ray = clip_outliers(gamma_ray, *clip_pct)

    params = fit_normalization(gamma_ray)
    gamma_ray_norm = params.apply(gamma_ray)
    return depth, gamma_ray_norm, params


def curve_to_frame(depth: np.ndarray, gamma_ray: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"depth": depth, "gamma_ray": gamma_ray})
