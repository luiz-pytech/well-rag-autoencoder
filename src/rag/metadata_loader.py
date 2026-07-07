"""Extração de metadados estruturados a partir do texto OCR do cabeçalho do perfil composto.

O texto vem do GLM-OCR (`pdf_ocr.ocr_header`), um modelo multimodal que entende a
estrutura de formulário do cabeçalho e devolve texto relativamente limpo, no
formato aproximado "RÓTULO: valor" por linha — mas não é perfeitamente
determinístico: a capitalização do rótulo varia (ex.: "BACIA:" vs "Bacia:"),
alguns campos aparecem colados na mesma linha que outro campo, e campos vazios no
documento original às vezes "vazam" o texto do próximo rótulo visível (ex.:
"ÁREA OU CAMPO:\nPROFUNDIDADE" quando a área/campo não está preenchida no PDF).
Os regex abaixo toleram essas variações; as validações de sanidade (formação
curta, profundidade plausível) continuam necessárias para o caso de vazamento.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class WellMetadata:
    well_id: str
    source_pdf: str
    nome_do_poco: str | None = None
    tipo_documento: str = "perfil_composto"
    bacia: str | None = None
    area_campo: str | None = None
    bloco: str | None = None
    profundidade_perfuracao_m: float | None = None
    profundidade_perfilagem_m: float | None = None
    formacao: str | None = None
    operador: str | None = None
    datum: str | None = None
    curvas_disponiveis: list[str] = field(default_factory=list)
    possui_coluna_litologica: bool = True  # todo perfil composto ANP tem essa seção no layout
    possui_gamma_ray: bool = False
    ocr_raw_text: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# Rótulos de seção conhecidos do template — se um campo vazio no documento original
# "vazar" o rótulo da próxima seção visível (comum quando ÁREA OU CAMPO ou similar
# não está preenchido no PDF), tratamos o valor capturado como não encontrado.
_KNOWN_SECTION_LABELS = {
    "PROFUNDIDADE", "ELEVAÇÕES", "ELEVACOES", "DATAS", "EQUIPAMENTO",
    "COORDENADAS", "LOCALIZAÇÃO", "LOCALIZACAO", "DESIGNAÇÃO", "DESIGNACAO",
    "COORDENADAS DO FUNDO", "COORDENADAS DA BOCA", "ESQUEMA DO POÇO",
}


def _clean_value(value: str) -> str:
    return value.strip(" :\t—-")


def _search(pattern: str, text: str, flags=re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    value = _clean_value(match.group(1))
    if not value or value.upper() in _KNOWN_SECTION_LABELS:
        return None
    return value


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[\d]+[.,]?\d*", value)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def parse_header_text(well_id: str, source_pdf: str, text: str) -> WellMetadata:
    """Faz o parsing do texto (GLM-OCR) do cabeçalho em campos estruturados."""

    # Nome do poço: geralmente aparece como "PETROBRAS [SUBMARINO] - <numero>" ou
    # "PETROBRAS SUBMARINO N.<numero>" logo após o rótulo "NOME DO POÇO ANP".
    nome_do_poco = _search(r"PETROBRAS(?:\s+SUBMARINO)?\s*(?:-|N\.)\s*(\d+)", text)
    if nome_do_poco:
        nome_do_poco = f"PETROBRAS - {nome_do_poco}"

    bacia = _search(r"BACIA\s*:?\s*([A-Za-zÀ-ú][A-Za-zÀ-ú0-9 ]*?(?:\([A-Za-zÀ-ú ]+\))?)(?:\s*\n|\s+PERFURA)", text)
    if bacia and any(w in bacia.upper() for w in ("COORDENADAS", "ALVO", "PROFUNDIDADE")):
        bacia = None

    area_campo = _search(r"[ÁA]REA\s+OU\s+CAMPO\s*:?\s*([A-Za-zÀ-ú][A-Za-zÀ-ú0-9 ]+?)(?:\s*\n|\s+PROFUNDIDADE)", text)

    bloco = _search(r"BLOCO\s*:?\s*([A-Z0-9\-]+)", text)

    formacao = _search(r"FORMA[ÇC][ÃA]O\s*:?\s*([A-Za-zÀ-ú][A-Za-zÀ-ú0-9 ]+?)(?:\s*\n|$)", text)
    # Nome de formação geológica é curto (1-2 palavras, raramente 3); valores
    # longos, ou contendo palavras de rótulos de outras seções, indicam vazamento.
    if formacao and (
        len(formacao.split()) > 3
        or "REVESTIMENTOS" in formacao.upper()
        or "CANHONEIOS" in formacao.upper()
        or "DATAS" in formacao.upper()
    ):
        formacao = None

    operador = _search(r"OPERADOR\s*:?\s*([A-ZÀ-Ú][A-ZÀ-Ú ]+?)(?:\s+PONTO|\s*\n|$)", text)

    datum = _search(r"DATUM\s*:?\s*([A-Z0-9\-]+)", text)

    perfuracao = _to_float(_search(r"PERFURA[ÇC][ÃA]O\s*:?\s*([\d.,]+)\s*m", text))
    perfilagem = _to_float(_search(r"PERFILAGEM\s*:?\s*([\d.,]+)\s*m", text))

    # Sanidade física: poços deste dataset (terrestres/marítimos rasos-a-médios da
    # ANP) não passam de ~10 km de profundidade — valores acima disso indicam erro
    # de leitura nos dígitos.
    MAX_PLAUSIBLE_DEPTH_M = 10000.0
    if perfuracao is not None and perfuracao > MAX_PLAUSIBLE_DEPTH_M:
        perfuracao = None
    if perfilagem is not None and perfilagem > MAX_PLAUSIBLE_DEPTH_M:
        perfilagem = None

    curvas_section = _search(r"PERFIS\s+CORRIDOS\s*:?\s*(.+?)(?:KOP|REVEST|$)", text, flags=re.IGNORECASE | re.DOTALL)
    curvas_disponiveis = []
    if curvas_section:
        curvas_disponiveis = sorted(set(re.findall(r"\b([A-Z]{2,6})\b\s*[—:-]?\s*\(", curvas_section)))

    possui_gamma_ray = bool(re.search(r"\bGR\b\s*[—:-]?\s*\(|\bGAMMA\s*RAY\b", text, re.IGNORECASE))

    return WellMetadata(
        well_id=well_id,
        source_pdf=source_pdf,
        nome_do_poco=nome_do_poco,
        bacia=bacia,
        area_campo=area_campo,
        bloco=bloco,
        profundidade_perfuracao_m=perfuracao,
        profundidade_perfilagem_m=perfilagem,
        formacao=formacao,
        operador=operador,
        datum=datum,
        curvas_disponiveis=curvas_disponiveis,
        possui_gamma_ray=possui_gamma_ray,
        ocr_raw_text=text,
    )


def save_metadata_list(metadata_list: list[WellMetadata], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in metadata_list], f, ensure_ascii=False, indent=2)


def load_metadata_list(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
