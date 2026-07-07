"""Geração de chunks textuais descritivos a partir dos metadados estruturados.

Os metadados exatos (bacia, profundidade, curvas...) já respondem perguntas de
campo exato diretamente via busca estruturada (ver retrieval.py). Os chunks
aqui servem para a busca vetorial/semântica: descrevem o documento em
linguagem natural, permitindo perguntas mais abertas ("o que esse perfil
mostra sobre a bacia X?", "quais poços têm coluna litológica?") que não batem
palavra-por-palavra com um campo estruturado.

Cada poço gera um chunk por "faceta" (identificação, profundidade/formação,
curvas disponíveis) em vez de um chunk único e longo — chunks menores e
temáticos tendem a ter embeddings mais específicos e melhoram o recall em
buscas pontuais.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    well_id: str
    text: str
    source: str  # de qual campo/seção do metadata este chunk foi derivado

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(value, default: str = "não informado(a)") -> str:
    if value is None or value == "":
        return default
    return str(value)


def build_chunks_for_well(meta: dict) -> list[Chunk]:
    """Gera os chunks descritivos de um único poço a partir do seu registro de metadados."""
    well_id = meta["well_id"]
    chunks = []

    identificacao = (
        f"O poço {well_id} (designação: {_fmt(meta.get('nome_do_poco'))}) é um documento do tipo "
        f"{_fmt(meta.get('tipo_documento'), 'perfil composto')}, operado por {_fmt(meta.get('operador'))}, "
        f"localizado no bloco {_fmt(meta.get('bloco'))}, na bacia {_fmt(meta.get('bacia'))}, "
        f"área/campo {_fmt(meta.get('area_campo'))}."
    )
    chunks.append(Chunk(chunk_id=f"{well_id}::identificacao", well_id=well_id, text=identificacao, source="identificacao"))

    perfuracao = meta.get("profundidade_perfuracao_m")
    perfilagem = meta.get("profundidade_perfilagem_m")
    formacao = meta.get("formacao")
    profundidade_texto = (
        f"O poço {well_id} tem profundidade de perfuração de "
        f"{_fmt(f'{perfuracao} metros' if perfuracao is not None else None)} e profundidade de perfilagem de "
        f"{_fmt(f'{perfilagem} metros' if perfilagem is not None else None)}. "
        f"A formação encontrada na profundidade final é {_fmt(formacao)}. Datum de referência: {_fmt(meta.get('datum'))}."
    )
    chunks.append(Chunk(chunk_id=f"{well_id}::profundidade", well_id=well_id, text=profundidade_texto, source="profundidade"))

    curvas = meta.get("curvas_disponiveis") or []
    curvas_texto = (
        f"O poço {well_id} possui as seguintes curvas de perfilagem corridas: "
        f"{', '.join(curvas) if curvas else 'nenhuma curva identificada nos metadados'}. "
        f"Este documento {'possui' if meta.get('possui_gamma_ray') else 'não possui indicação de'} curva Gamma Ray (GR). "
        f"O perfil composto {'inclui' if meta.get('possui_coluna_litologica') else 'não inclui'} coluna litológica."
    )
    chunks.append(Chunk(chunk_id=f"{well_id}::curvas", well_id=well_id, text=curvas_texto, source="curvas"))

    return chunks


def build_all_chunks(metadata_list: list[dict]) -> list[Chunk]:
    chunks = []
    for meta in metadata_list:
        chunks.extend(build_chunks_for_well(meta))
    return chunks


def save_chunks(chunks: list[Chunk], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[dict]:
    chunks = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
