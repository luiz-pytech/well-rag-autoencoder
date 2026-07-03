"""Métricas de reconstrução e baselines para comparação com o Autoencoder 1D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ReconstructionScores:
    mse: float
    mae: float
    per_window_error: np.ndarray  # MSE por janela


def score_reconstruction(original: np.ndarray, reconstructed: np.ndarray) -> ReconstructionScores:
    """original/reconstructed: shape (N, window_size)."""
    mse = mean_squared_error(original, reconstructed)
    mae = mean_absolute_error(original, reconstructed)
    per_window = np.mean((original - reconstructed) ** 2, axis=1)
    return ReconstructionScores(mse=mse, mae=mae, per_window_error=per_window)


# --------------------------------------------------------------------------
# Baseline 1: PCA
# --------------------------------------------------------------------------

def pca_baseline(train_windows: np.ndarray, test_windows: np.ndarray, n_components: int = 8) -> np.ndarray:
    """Reconstrói janelas de teste projetando/reprojetando via PCA treinado no conjunto de treino."""
    pca = PCA(n_components=n_components)
    pca.fit(train_windows)
    latent = pca.transform(test_windows)
    return pca.inverse_transform(latent)


# --------------------------------------------------------------------------
# Baseline 2: média móvel (suavização como "reconstrução")
# --------------------------------------------------------------------------

def moving_average_baseline(windows: np.ndarray, window_length: int = 9) -> np.ndarray:
    """Aplica média móvel centrada em cada janela como reconstrução ingênua."""
    if window_length % 2 == 0:
        window_length += 1
    pad = window_length // 2
    kernel = np.ones(window_length) / window_length

    smoothed = np.empty_like(windows)
    for i, row in enumerate(windows):
        padded = np.pad(row, pad, mode="edge")
        smoothed[i] = np.convolve(padded, kernel, mode="valid")
    return smoothed


# --------------------------------------------------------------------------
# Baseline 3: Isolation Forest sobre features estatísticas das janelas
# --------------------------------------------------------------------------

def extract_window_features(windows: np.ndarray) -> np.ndarray:
    """Features simples por janela: média, desvio padrão, mín, máx, inclinação e energia."""
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    minimum = windows.min(axis=1)
    maximum = windows.max(axis=1)
    energy = np.mean(windows ** 2, axis=1)

    x = np.arange(windows.shape[1])
    slope = np.array([np.polyfit(x, row, 1)[0] for row in windows])

    return np.column_stack([mean, std, minimum, maximum, slope, energy])


def isolation_forest_scores(
    train_windows: np.ndarray, test_windows: np.ndarray, contamination: float = 0.05, seed: int = 42
) -> np.ndarray:
    """Retorna um score de anomalia por janela de teste (maior = mais anômalo)."""
    train_feats = extract_window_features(train_windows)
    test_feats = extract_window_features(test_windows)

    clf = IsolationForest(contamination=contamination, random_state=seed)
    clf.fit(train_feats)
    # decision_function: quanto menor (mais negativo), mais anômalo -> inverte o sinal
    return -clf.decision_function(test_feats)
