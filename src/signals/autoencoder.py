"""Autoencoder 1D (Conv1D) para reconstrução de janelas da curva Gamma Ray."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Conv1DAutoencoder(nn.Module):
    """Autoencoder convolucional simples para séries 1D.

    Espera entrada com shape (batch, 1, window_size). window_size deve ser
    divisível por 4 (duas camadas de stride=2 no encoder/decoder).
    """

    def __init__(self, window_size: int = 128, latent_channels: int = 16):
        super().__init__()
        if window_size % 4 != 0:
            raise ValueError("window_size deve ser divisível por 4.")

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, latent_channels, kernel_size=3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def _to_tensor(windows: np.ndarray) -> torch.Tensor:
    return torch.tensor(windows, dtype=torch.float32).unsqueeze(1)  # (N, 1, window_size)


def train_autoencoder(
    model: Conv1DAutoencoder,
    train_windows: np.ndarray,
    val_windows: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
    verbose: bool = True,
) -> TrainHistory:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = TensorDataset(_to_tensor(train_windows))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_tensor = _to_tensor(val_windows).to(device) if val_windows is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = TrainHistory()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)

        train_loss = running_loss / len(train_ds)
        history.train_loss.append(train_loss)

        val_msg = ""
        if val_tensor is not None:
            model.eval()
            with torch.no_grad():
                val_recon = model(val_tensor)
                val_loss = criterion(val_recon, val_tensor).item()
            history.val_loss.append(val_loss)
            val_msg = f" | val_loss={val_loss:.5f}"

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"[epoch {epoch:03d}/{epochs}] train_loss={train_loss:.5f}{val_msg}")

    return history


@torch.no_grad()
def reconstruct(model: Conv1DAutoencoder, windows: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    x = _to_tensor(windows).to(device)
    recon = model(x)
    return recon.squeeze(1).cpu().numpy()


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Erro de reconstrução (MSE) por janela. Shape de entrada: (N, window_size)."""
    return np.mean((original - reconstructed) ** 2, axis=1)
