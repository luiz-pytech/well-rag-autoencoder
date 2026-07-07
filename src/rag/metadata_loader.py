"""Extração de metadados estruturados a partir do texto OCR do cabeçalho do perfil composto.

O layout do cabeçalho é padronizado (mesmo template Petrobras/ANP em todos os
poços), mas o OCR introduz ruído (acentos viram '�', espaçamento irregular).
Os regex abaixo são tolerantes a isso: usam âncoras de rótulo (ex.: "BACIA")
sem depender de acentuação correta, e capturam o valor até a próxima quebra de
linha ou próximo rótulo conhecido.
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


# Rótulos de seção conhecidos do template — se um regex de campo "vazar" e capturar
# um desses como valor (comum quando o campo real está vazio/ilegível no OCR e o
# rótulo da seção seguinte aparece logo em seguida no texto), tratamos como não
# encontrado em vez de propagar um valor obviamente errado.
_KNOWN_SECTION_LABELS = {
    "PROFUNDIDADE", "ELEVAÇÕES", "ELEVACOES", "DATAS", "EQUIPAMENTO",
    "COORDENADAS", "LOCALIZAÇÃO", "LOCALIZACAO", "DESIGNAÇÃO", "DESIGNACAO",
}


def _search(pattern: str, text: str, flags=re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    value = match.group(1).strip(" :\n\t")
    if value.upper() in _KNOWN_SECTION_LABELS:
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
    """Faz o parsing do texto OCR do cabeçalho em campos estruturados.

    O OCR erra alguns caracteres em nomes curtos de curva (ex.: "AIT" -> "Am",
    "GR" -> "R", "FMI" -> "FM"), então `curvas_disponiveis`/`possui_gamma_ray`
    são heurísticos e podem ter falsos negativos — ver limitação documentada
    no README/discussão. Os campos numéricos (profundidades) e nomes longos
    (bacia, formação) são mais confiáveis.
    """
    nome_do_poco = _search(r"PETROBRAS\s*-\s*(\d+)", text) or _search(r"DESIGNA[�A][�C][�A]O\s*\n?\s*([A-Z0-9\- ]+)", text)
    if nome_do_poco and nome_do_poco.isdigit():
        nome_do_poco = f"PETROBRAS - {nome_do_poco}"

    # OCR normalmente reconhece acentos corretamente (Ç, Ã, Á...); o caractere de
    # replacement (U+FFFD) só aparece em casos raros de glifo irreconhecível, então
    # os regex toleram AMBOS: acento real OU falha de decodificação.
    NOISE = "�"

    bacia = _search(rf"BACIA\s*:?\s*([A-ZÀ-Ú{NOISE}][A-Za-zÀ-Ú{NOISE} ]*?(?:\([A-Za-z ]+\))?)(?:\s+PERFURA|\n)", text)
    # Palavras de rótulos de outras seções que ocasionalmente vazam para o valor
    # de bacia (mesma causa raiz dos vazamentos de area_campo/formacao).
    if bacia and any(w in bacia.upper() for w in ("COORDENADAS", "ALVO", "PROFUNDIDADE")):
        bacia = None
    area_campo = _search(rf"[AÁ{NOISE}]REA\s+OU\s+CAMPO\s*:?\s*([A-ZÀ-Ú{NOISE}][A-Za-zÀ-Ú{NOISE} ]+?)(?:\s+PROFUNDIDADE|\n)", text)
    bloco = _search(r"BLOCO\s*:?\s*([A-Z0-9\-]+)", text)
    formacao = _search(rf"FORMA[ÇC{NOISE}][ÃA{NOISE}]O\s*:?\s*\n?\s*[{NOISE}:—-]?\s*([A-ZÀ-Ú{NOISE}][A-Za-zÀ-Ú{NOISE} ]+?)(?:\n|$)", text)
    # Nome de formação geológica é curto (1-2 palavras, raramente 3); valores
    # longos, ou contendo palavras de rótulos de seção conhecidas, indicam
    # vazamento para o rótulo da próxima seção do template (mesma causa do
    # vazamento em area_campo, mas o rótulo seguinte varia por layout do PDF).
    if formacao and (len(formacao.split()) > 3 or "REVESTIMENTOS" in formacao.upper() or "CANHONEIOS" in formacao.upper()):
        formacao = None
    operador = _search(rf"OPERADOR\s*:?\s*([A-Z{NOISE}]+)", text)
    datum = _search(r"DATUM\s*:?\s*([A-Z0-9\-]+)", text)

    perfuracao = _to_float(_search(rf"PERFURA[ÇC{NOISE}][ÃA{NOISE}]O\s*:?\s*([\d.,]+)\s*m", text))
    perfilagem = _to_float(_search(r"PERFILAGEM\s*:?\s*([\d.,]+)\s*m", text))

    # Sanidade física: poços deste dataset (terrestres/marítimos rasos-a-médios da
    # ANP) não passam de ~10 km de profundidade — valores acima disso indicam erro
    # de OCR nos dígitos (dígito duplicado, ponto decimal deslocado etc.).
    MAX_PLAUSIBLE_DEPTH_M = 10000.0
    if perfuracao is not None and perfuracao > MAX_PLAUSIBLE_DEPTH_M:
        perfuracao = None
    if perfilagem is not None and perfilagem > MAX_PLAUSIBLE_DEPTH_M:
        perfilagem = None

    curvas_section = _search(r"PERFIS\s+CORRIDOS\s*:?\s*(.+?)(?:KOP|REVEST|$)", text, flags=re.IGNORECASE | re.DOTALL)
    curvas_disponiveis = []
    if curvas_section:
        # mnemônicos de 2-6 letras maiúsculas antes de um parêntese de profundidade,
        # tolerando espaço/ruído entre a sigla e o parêntese (ex.: "CMR — (1000.0")
        curvas_disponiveis = sorted(set(re.findall(rf"\b([A-Z]{{2,6}})\b\s*[{NOISE}:—-]?\s*\(", curvas_section)))

    possui_gamma_ray = bool(re.search(rf"\bGR\b\s*[{NOISE}:—-]?\s*\(|\bR\s*\(\s*15", text))

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
