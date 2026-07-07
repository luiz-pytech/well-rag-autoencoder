"""Pipeline de perguntas e respostas: classifica a pergunta, recupera evidência
(metadados estruturados e/ou chunks vetoriais) e gera a resposta final.

Geração: usa a API da OpenAI (GPT) quando OPENAI_API_KEY está configurada.
Sem chave, cai em um fallback por template que monta a resposta diretamente a
partir da evidência recuperada — sem inventar nada, só sem prosa natural do LLM.
Isso permite que o pipeline funcione (e seja avaliado) mesmo antes de configurar
a chave de API.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Carrega OPENAI_API_KEY do arquivo .env na raiz do projeto, se existir. Não
# sobrescreve uma variável de ambiente já definida no processo (override=False).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass

SYSTEM_PROMPT = (
    "Você é um assistente técnico para consulta de perfis convencionais de poços. "
    "Responda apenas com base nos metadados e trechos fornecidos. Se a informação "
    "não estiver presente, diga que a informação não foi encontrada. Não invente "
    "profundidades, curvas, formações, coordenadas ou interpretações petrofísicas. "
    "Sempre apresente a evidência usada."
)

MODEL_ID = "gpt-4o-mini"

# Campos exatos reconhecidos para consulta estruturada direta (pergunta -> campo do metadata.json)
FIELD_QUESTIONS = {
    "nome_do_poco": [r"nome do po[çc]o"],
    "bacia": [r"\bbacia\b"],
    "area_campo": [r"\b[aá]rea\b", r"\bcampo\b"],
    "bloco": [r"\bbloco\b"],
    "profundidade_perfuracao_m": [r"profundidade.*perfura[çc][ãa]o", r"perfura[çc][ãa]o.*profundidade"],
    "profundidade_perfilagem_m": [r"profundidade.*perfilagem", r"perfilagem.*profundidade"],
    "formacao": [r"\bforma[çc][ãa]o\b"],
    "operador": [r"\boperador\b"],
    "curvas_disponiveis": [r"curvas dispon[íi]veis", r"quais curvas", r"perfis corridos"],
    "possui_coluna_litologica": [r"coluna litol[óo]gica"],
    "possui_gamma_ray": [r"gamma ray", r"\bgr\b"],
    "tipo_documento": [r"tipo de documento"],
}


@dataclass
class QAResult:
    question: str
    answer: str
    evidence: list[dict]
    method: str  # "structured", "vector", "hybrid"


def extract_well_id(question: str, known_well_ids: list[str]) -> str | None:
    """Tenta identificar a qual poço a pergunta se refere, buscando o well_id na pergunta."""
    q_upper = question.upper()
    for well_id in known_well_ids:
        if well_id.upper() in q_upper:
            return well_id
    return None


def detect_exact_field(question: str) -> str | None:
    """Detecta se a pergunta pede um campo estruturado exato conhecido."""
    q_lower = question.lower()
    for field, patterns in FIELD_QUESTIONS.items():
        if any(re.search(p, q_lower) for p in patterns):
            return field
    return None


def _format_field_value(field: str, value) -> str:
    if value is None:
        return "informação não encontrada nos metadados"
    if field == "curvas_disponiveis":
        return ", ".join(value) if value else "nenhuma curva identificada"
    if field in ("possui_coluna_litologica", "possui_gamma_ray"):
        return "sim" if value else "não"
    if field in ("profundidade_perfuracao_m", "profundidade_perfilagem_m"):
        return f"{value} metros"
    return str(value)


def answer_structured(well_meta: dict, field: str) -> QAResult:
    """Responde uma pergunta de campo exato diretamente dos metadados — sem LLM."""
    value = well_meta.get(field)
    formatted = _format_field_value(field, value)
    answer = f"{field.replace('_', ' ').capitalize()}: {formatted}. Fonte: metadados extraídos de {well_meta.get('source_pdf', 'documento não identificado')}."
    return QAResult(
        question="",
        answer=answer,
        evidence=[{"type": "structured_metadata", "well_id": well_meta["well_id"], "field": field, "value": value}],
        method="structured",
    )


def _template_answer_from_chunks(question: str, chunks: list[dict]) -> str:
    """Fallback sem LLM: concatena os chunks recuperados como evidência bruta."""
    if not chunks:
        return "A informação não foi encontrada nos documentos disponíveis."
    evidence_text = "\n".join(f"- [{c['well_id']}] {c['text']}" for c in chunks)
    return f"Com base nos trechos recuperados:\n{evidence_text}"


def _generate_with_openai(question: str, evidence_text: str) -> str | None:
    """Gera a resposta final via API da OpenAI.

    Retorna None (caindo no fallback por template) se a chave não estiver
    configurada, for inválida/placeholder, ou se a chamada falhar por qualquer
    motivo de rede/autenticação — o pipeline nunca deve quebrar por causa da
    geração opcional via LLM.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    import openai

    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Pergunta: {question}\n\nEvidência disponível:\n{evidence_text}"},
            ],
        )
    except openai.APIError:
        return None
    return response.choices[0].message.content


def answer_with_chunks(question: str, chunks: list[dict]) -> QAResult:
    """Responde uma pergunta semântica/aberta usando os chunks recuperados como evidência."""
    evidence_text = "\n".join(f"[{c['well_id']}] {c['text']}" for c in chunks) if chunks else "(nenhum trecho relevante encontrado)"

    answer = _generate_with_openai(question, evidence_text)
    method = "vector"
    if answer is None:
        answer = _template_answer_from_chunks(question, chunks)
        method = "vector_template_fallback"

    return QAResult(
        question=question,
        answer=answer,
        evidence=[{"type": "chunk", **c} for c in chunks],
        method=method,
    )
