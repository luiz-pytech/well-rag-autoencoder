"""Gráficos para análise da curva Gamma Ray e do erro de reconstrução."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_curve(depth: np.ndarray, gamma_ray: np.ndarray, title: str = "Gamma Ray", ax=None):
    """Plota a curva GR x profundidade no estilo tradicional de perfil (profundidade no eixo Y, invertida)."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(3, 8))

    ax.plot(gamma_ray, depth, color="green", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Gamma Ray")
    ax.set_ylabel("Profundidade")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if created:
        plt.tight_layout()
    return ax


def plot_original_vs_reconstructed(depth: np.ndarray, original: np.ndarray, reconstructed: np.ndarray):
    """Compara curva original e reconstruída lado a lado (mesma escala de profundidade)."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
    plot_curve(depth, original, title="Original", ax=axes[0])
    plot_curve(depth, reconstructed, title="Reconstruída", ax=axes[1])
    plt.tight_layout()
    return fig


def plot_reconstruction_error(depth: np.ndarray, error: np.ndarray, threshold: float | None = None):
    """Erro de reconstrução por profundidade, com destaque para trechos acima do limiar."""
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.plot(error, depth, color="crimson", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Erro de reconstrução (MSE)")
    ax.set_ylabel("Profundidade")
    ax.set_title("Erro de reconstrução")
    ax.grid(alpha=0.3)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=0.8, label=f"limiar={threshold:.3f}")
        anomalous = error >= threshold
        ax.scatter(error[anomalous], depth[anomalous], color="black", s=8, zorder=3, label="anômalo")
        ax.legend()

    plt.tight_layout()
    return fig


def plot_error_distribution(error: np.ndarray, bins: int = 30):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(error, bins=bins, color="steelblue", edgecolor="white")
    ax.set_xlabel("Erro de reconstrução (MSE por janela)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição do erro de reconstrução")
    plt.tight_layout()
    return fig


def plot_method_comparison(depth: np.ndarray, errors_by_method: dict[str, np.ndarray]):
    """Compara o erro de reconstrução (ou score de anomalia) de múltiplos métodos por profundidade."""
    fig, ax = plt.subplots(figsize=(5, 8))
    for name, error in errors_by_method.items():
        ax.plot(error, depth, linewidth=0.8, label=name)
    ax.invert_yaxis()
    ax.set_xlabel("Erro / score de anomalia")
    ax.set_ylabel("Profundidade")
    ax.set_title("Comparação entre métodos")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
