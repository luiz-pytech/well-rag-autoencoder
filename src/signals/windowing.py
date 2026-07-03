"""Divisão da curva Gamma Ray em janelas fixas e split treino/teste."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Windows:
    data: np.ndarray  # shape (n_windows, window_size)
    start_depths: np.ndarray  # profundidade inicial de cada janela
    window_size: int
    stride: int


def make_windows(depth: np.ndarray, values: np.ndarray, window_size: int = 128, stride: int | None = None) -> Windows:
    """Fatia a curva em janelas de tamanho fixo com stride configurável.

    stride=None equivale a janelas não sobrepostas (stride = window_size).
    """
    if stride is None:
        stride = window_size

    n = len(values)
    starts = np.arange(0, n - window_size + 1, stride)
    if len(starts) == 0:
        raise ValueError(f"Curva com {n} pontos é menor que window_size={window_size}.")

    data = np.stack([values[s : s + window_size] for s in starts])
    start_depths = depth[starts]
    return Windows(data=data, start_depths=start_depths, window_size=window_size, stride=stride)


def train_test_split_windows(
    windows: Windows, test_size: float = 0.2, shuffle: bool = True, seed: int = 42
) -> tuple[Windows, Windows]:
    """Split treino/teste. Por padrão embaralha (janelas são amostras ~i.i.d. para o AE);
    use shuffle=False se quiser preservar a ordem em profundidade (ex.: avaliar um trecho contíguo)."""
    n = windows.data.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_test = max(1, int(n * test_size))
    test_idx = np.sort(idx[:n_test]) if not shuffle else idx[:n_test]
    train_idx = np.sort(idx[n_test:]) if not shuffle else idx[n_test:]

    train = Windows(
        data=windows.data[train_idx],
        start_depths=windows.start_depths[train_idx],
        window_size=windows.window_size,
        stride=windows.stride,
    )
    test = Windows(
        data=windows.data[test_idx],
        start_depths=windows.start_depths[test_idx],
        window_size=windows.window_size,
        stride=windows.stride,
    )
    return train, test
