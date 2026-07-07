"""Carregamento em lote de múltiplos poços para treino unificado do Autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gamma_ray_loader import load_gamma_ray
from .preprocessing import preprocess_curve
from .windowing import Windows, make_windows


@dataclass
class WellWindows:
    well_id: str
    windows: Windows


def list_well_ids(gamma_ray_dir: Path) -> list[str]:
    return sorted({p.parent.parent.name for p in gamma_ray_dir.rglob("*.dlis")})


def load_well_windows(
    gamma_ray_dir: Path,
    well_id: str,
    window_size: int = 64,
    stride: int = 32,
) -> Windows | None:
    """Carrega, pré-processa e janela a curva GR de um poço.

    Retorna None se nenhum .dlis do poço tiver um canal GR convencional utilizável
    (ver gamma_ray_loader.load_from_dlis) ou se a curva resultante não tiver janelas
    suficientes — esses poços devem ser pulados pelo chamador, não tratados como erro.
    """
    dlis_files = sorted((gamma_ray_dir / well_id).rglob("*.dlis"))

    curve = None
    for fp in dlis_files:
        try:
            curve = load_gamma_ray(fp)
            break
        except ValueError:
            continue
    if curve is None:
        return None

    try:
        depth, gr_norm, _ = preprocess_curve(curve.depth, curve.gamma_ray)
        windows = make_windows(depth, gr_norm, window_size=window_size, stride=stride)
    except ValueError:
        return None

    if windows.data.shape[0] < 10:
        return None
    return windows


def load_all_wells(
    gamma_ray_dir: Path,
    window_size: int = 64,
    stride: int = 32,
    well_ids: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Carrega janelas de todos os poços (ou de `well_ids`, se informado).

    Retorna (dict well_id -> array de janelas (N, window_size), lista de well_ids pulados).
    """
    if well_ids is None:
        well_ids = list_well_ids(gamma_ray_dir)

    result = {}
    skipped = []
    for well_id in well_ids:
        windows = load_well_windows(gamma_ray_dir, well_id, window_size=window_size, stride=stride)
        if windows is None:
            skipped.append(well_id)
            continue
        result[well_id] = windows.data.astype(np.float32)

    return result, skipped


def split_wells(
    well_ids: list[str], test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Divide poços (não janelas) em treino/teste, para avaliar generalização real."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(well_ids)
    n_test = max(1, int(len(well_ids) * test_fraction))
    test_wells = sorted(shuffled[:n_test])
    train_wells = sorted(shuffled[n_test:])
    return train_wells, test_wells
