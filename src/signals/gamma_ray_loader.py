"""Leitura da curva Gamma Ray a partir de arquivos DLIS ou CSV.

DLIS armazena curvas por frame (conjunto de canais amostrados na mesma grade de
profundidade/tempo). Um .dlis pode ter vários logical files e frames (uma por
corrida de ferramenta); a profundidade de cada frame é obtida via `frame.index`
(dlisio já resolve isso, independente do nome do canal — na prática mnemônicos
como "INDEX023" são comuns). O canal de Gamma Ray é identificado por um dos
mnemônicos usuais (GR, GRC, SGR, CGR, ECGR...).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GAMMA_RAY_MNEMONICS = ("GR", "GRC", "SGR", "CGR", "ECGR", "GRD", "HGR")

# Espaçamento típico de amostragem de um perfil Gamma Ray convencional (em metros).
# Ferramentas de imagem/alta resolução (ex.: CAST, acústica) amostram na casa de
# milímetros e às vezes carregam um canal auxiliar com nome parecido com GR — sem
# esse filtro, o critério de "mais amostras válidas" as escolhe por engano, já que
# naturalmente têm muito mais pontos no mesmo intervalo de profundidade.
MIN_DEPTH_STEP_M = 0.02
MAX_DEPTH_STEP_M = 0.60

# Fatores de conversão para metros. DLIS registra a unidade real da ferramenta no
# canal de índice (ex.: "m", "0.1 in", "ft") — não dá para assumir metros direto.
_DEPTH_UNIT_TO_METERS = {
    "m": 1.0,
    "in": 0.0254,
    "0.1 in": 0.00254,
    "ft": 0.3048,
}


def _depth_to_meters(depth: np.ndarray, unit: str | None) -> tuple[np.ndarray, float]:
    unit_key = (unit or "m").strip().lower()
    factor = _DEPTH_UNIT_TO_METERS.get(unit_key)
    if factor is None:
        raise ValueError(
            f"Unidade de profundidade desconhecida: {unit!r}. "
            f"Adicione o fator de conversão em _DEPTH_UNIT_TO_METERS."
        )
    return depth * factor, factor


@dataclass
class GammaRayCurve:
    depth: np.ndarray
    gamma_ray: np.ndarray
    well_name: str
    source_file: str
    depth_unit: str = "m"
    gamma_ray_unit: str = "gAPI"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"depth": self.depth, "gamma_ray": self.gamma_ray})


def _pick_mnemonic(available: list[str], candidates: tuple[str, ...]) -> str | None:
    upper_map = {name.upper(): name for name in available}
    for candidate in candidates:
        if candidate in upper_map:
            return upper_map[candidate]
    return None


def load_from_dlis(path: str | Path) -> GammaRayCurve:
    """Extrai a curva Gamma Ray de um arquivo .dlis usando dlisio.

    Um .dlis pode conter vários "logical files" (uma por corrida de ferramenta) e
    vários frames, cada um com seu próprio canal de índice (profundidade) — o nome
    desse canal não segue um mnemônico fixo (ex.: "INDEX023", "INDEX025"...), então
    usamos `frame.index` (fornecido pelo dlisio) em vez de tentar adivinhar por nome.

    Quando há mais de um frame com um canal Gamma Ray, descarta os que têm espaçamento
    de amostragem fora da faixa típica de um GR convencional (MIN/MAX_DEPTH_STEP_M) —
    isso evita escolher por engano ferramentas de imagem/alta resolução (ex.: CAST,
    acústica) que amostram na casa de milímetros e às vezes carregam um canal auxiliar
    com nome parecido com GR. Entre os candidatos restantes, escolhe o de maior número
    de amostras válidas (mais provável de ser a corrida principal/mais completa).
    """
    try:
        from dlisio import dlis
    except ImportError as exc:
        raise ImportError(
            "dlisio não instalado. Rode `pip install dlisio` para ler arquivos .dlis."
        ) from exc

    path = Path(path)
    best: GammaRayCurve | None = None
    best_n_valid = -1
    last_error: Exception | None = None
    rejected_sampling: list[float] = []

    try:
        dlis_handle = dlis.load(str(path))
    except Exception as exc:
        raise ValueError(f"Não foi possível abrir {path} como DLIS (arquivo corrompido ou formato não suportado): {exc}") from exc

    with dlis_handle as files:
        for logical_file in files:
            for frame in logical_file.frames:
                channel_names = [ch.name for ch in frame.channels]
                gr_name = _pick_mnemonic(channel_names, GAMMA_RAY_MNEMONICS)
                if gr_name is None:
                    continue

                try:
                    curves = frame.curves()
                    depth_raw = np.asarray(curves[frame.index], dtype=float).ravel()
                    gamma_ray = np.asarray(curves[gr_name], dtype=float).ravel()

                    index_channel = next(ch for ch in frame.channels if ch.name == frame.index)
                    depth, _ = _depth_to_meters(depth_raw, index_channel.units)
                except Exception as exc:  # curva corrompida, incompatível ou unidade desconhecida
                    last_error = exc
                    continue

                depth_sorted = np.sort(depth)
                step = float(np.median(np.diff(depth_sorted))) if len(depth_sorted) > 1 else 0.0
                if not (MIN_DEPTH_STEP_M <= step <= MAX_DEPTH_STEP_M):
                    rejected_sampling.append(step)
                    continue

                n_valid = int(np.sum(~np.isnan(gamma_ray) & (gamma_ray > -999.0)))
                if n_valid <= best_n_valid:
                    continue

                well_name = getattr(logical_file.origins[0], "well_name", None) if logical_file.origins else None
                best_n_valid = n_valid
                best = GammaRayCurve(
                    depth=depth,
                    gamma_ray=gamma_ray,
                    well_name=well_name or path.stem,
                    source_file=str(path),
                )

    if best is None:
        detail = f" Último erro: {last_error}" if last_error else ""
        if rejected_sampling:
            detail += (
                f" {len(rejected_sampling)} frame(s) com canal GR-like descartado(s) por "
                f"amostragem fora de [{MIN_DEPTH_STEP_M}, {MAX_DEPTH_STEP_M}] m "
                f"(passos encontrados: {[round(s, 4) for s in rejected_sampling]})."
            )
        raise ValueError(
            f"Nenhum frame com canal Gamma Ray convencional encontrado em {path}.{detail}"
        )
    return best


def load_from_csv(path: str | Path, depth_col: str = "depth", gr_col: str = "gamma_ray") -> GammaRayCurve:
    """Lê a curva Gamma Ray de um CSV com colunas depth,gamma_ray."""
    path = Path(path)
    df = pd.read_csv(path)
    if depth_col not in df.columns or gr_col not in df.columns:
        raise ValueError(f"CSV {path} precisa ter as colunas '{depth_col}' e '{gr_col}'.")
    return GammaRayCurve(
        depth=df[depth_col].to_numpy(dtype=float),
        gamma_ray=df[gr_col].to_numpy(dtype=float),
        well_name=path.stem,
        source_file=str(path),
    )


def load_gamma_ray(path: str | Path) -> GammaRayCurve:
    """Detecta a extensão do arquivo e despacha para o loader adequado."""
    path = Path(path)
    if path.suffix.lower() == ".dlis":
        return load_from_dlis(path)
    if path.suffix.lower() == ".csv":
        return load_from_csv(path)
    raise ValueError(f"Extensão não suportada: {path.suffix}. Use .dlis ou .csv.")
