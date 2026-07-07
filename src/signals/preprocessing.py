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


def largest_valid_segment(depth: np.ndarray, gamma_ray: np.ndarray, max_gap_fraction: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Recorta o maior trecho contíguo sem grandes lacunas de dado ausente.

    Interpolar sobre uma lacuna enorme (ex.: 90% da curva sem leitura, comum em .dlis
    onde o canal GR só foi gravado num trecho curto do poço) cria um segmento
    artificialmente "liso" — uma reta entre os dois extremos válidos — que qualquer
    método de reconstrução acerta trivialmente, inflando métricas de forma enganosa.
    Em vez de interpolar cegamente, isolamos o maior bloco contíguo de amostras
    válidas cujas lacunas internas não passem de `max_gap_fraction` do próprio bloco.
    """
    valid = (gamma_ray > -999.0) & ~np.isnan(gamma_ray)
    if valid.sum() < 2:
        raise ValueError("Curva com menos de 2 pontos válidos.")

    # encontra os blocos contíguos de índices válidos, tratando pequenas lacunas
    # (< max_gap_fraction do tamanho já acumulado do bloco) como parte do mesmo bloco
    segments = []
    seg_start = None
    gap_run = 0
    seg_valid_count = 0

    for i, v in enumerate(valid):
        if v:
            if seg_start is None:
                seg_start = i
                seg_valid_count = 0
            seg_valid_count += 1
            gap_run = 0
        else:
            if seg_start is not None:
                gap_run += 1
                allowed_gap = max(1, int(seg_valid_count * max_gap_fraction))
                if gap_run > allowed_gap:
                    segments.append((seg_start, i - gap_run))
                    seg_start = None
                    gap_run = 0
                    seg_valid_count = 0

    if seg_start is not None:
        segments.append((seg_start, len(valid) - gap_run))

    if not segments:
        raise ValueError("Nenhum trecho contíguo de dado válido encontrado.")

    best_start, best_end = max(segments, key=lambda s: s[1] - s[0])
    return depth[best_start:best_end], gamma_ray[best_start:best_end]


def interpolate_missing(depth: np.ndarray, gamma_ray: np.ndarray) -> np.ndarray:
    """Interpola valores ausentes (NaN) e inválidos (ex.: -999.25) linearmente."""
    gr = gamma_ray.copy()
    gr[gr <= -999.0] = np.nan

    valid = ~np.isnan(gr)
    if valid.sum() < 2:
        raise ValueError("Curva com menos de 2 pontos válidos; não é possível interpolar.")

    gr[~valid] = np.interp(depth[~valid], depth[valid], gr[valid])
    return gr


def clip_outliers(gamma_ray: np.ndarray, low_pct: float = 0.1, high_pct: float = 99.9) -> np.ndarray:
    """Limita valores extremos (ex.: erros de ferramenta) aos percentis informados.

    Percentis mais próximos de 0/100 (0.1/99.9 em vez de 0.5/99.5) evitam achatar
    zonas reais de GR alto/baixo (ex.: folhelho radioativo genuíno cobrindo vários
    metros) no mesmo valor-teto — um clipping mais agressivo cria platôs artificiais
    que qualquer método reconstrói trivialmente, inflando as métricas sem motivo real.
    """
    low, high = np.percentile(gamma_ray, [low_pct, high_pct])
    return np.clip(gamma_ray, low, high)


def fit_normalization(gamma_ray: np.ndarray) -> NormalizationParams:
    return NormalizationParams(mean=float(np.mean(gamma_ray)), std=float(np.std(gamma_ray) + 1e-8))


def preprocess_curve(
    depth: np.ndarray,
    gamma_ray: np.ndarray,
    clip_pct: tuple[float, float] = (0.1, 99.9),
    max_missing_fraction: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, NormalizationParams]:
    """Pipeline completo: ordena, remove duplicatas, recorta o maior trecho válido,
    interpola pequenas lacunas internas, faz clipping e normaliza.

    Levanta ValueError se, mesmo após recortar o melhor trecho contíguo, a fração de
    dado ausente ainda exceder `max_missing_fraction` — arquivos DLIS onde o canal GR
    só foi gravado numa fração pequena do poço não devem virar retas interpoladas
    artificiais (ver largest_valid_segment).

    Retorna (depth_limpo, gamma_ray_normalizado, params_normalizacao).
    """
    depth, gamma_ray = sort_by_depth(depth, gamma_ray)
    depth, gamma_ray = drop_duplicate_depths(depth, gamma_ray)
    depth, gamma_ray = largest_valid_segment(depth, gamma_ray)

    missing_fraction = float(np.mean((gamma_ray <= -999.0) | np.isnan(gamma_ray)))
    if missing_fraction > max_missing_fraction:
        raise ValueError(
            f"Mesmo após recortar o maior trecho contíguo, {missing_fraction:.1%} dos pontos "
            f"estão ausentes (limite: {max_missing_fraction:.0%}). Curva de qualidade insuficiente."
        )

    gamma_ray = interpolate_missing(depth, gamma_ray)
    gamma_ray = clip_outliers(gamma_ray, *clip_pct)

    params = fit_normalization(gamma_ray)
    gamma_ray_norm = params.apply(gamma_ray)
    return depth, gamma_ray_norm, params


def curve_to_frame(depth: np.ndarray, gamma_ray: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"depth": depth, "gamma_ray": gamma_ray})
