"""Métricas de avaliação do RAG: recuperação (Recall@k, MRR, nDCG) e resposta
(acurácia em campo exato, taxa de resposta correta/parcial, taxa de alucinação).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Métricas de recuperação
# --------------------------------------------------------------------------
#
# O retriever opera em nível de CHUNK (cada poço gera vários chunks), mas a
# unidade de relevância é o POÇO. Sem deduplicar, um poço relevante que ocupa
# 3 das top-5 posições infla o DCG acima do IDCG (que assume 1 item relevante
# por posição), estourando nDCG > 1. `_dedupe_by_first_occurrence` reduz a
# lista de chunks recuperados a well_ids únicos, mantendo a primeira (melhor)
# posição de cada poço — assim todas as métricas abaixo operam sobre "poços
# recuperados", não "chunks recuperados".


def _dedupe_by_first_occurrence(well_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for well_id in well_ids:
        if well_id not in seen:
            seen.add(well_id)
            deduped.append(well_id)
    return deduped


def recall_at_k(retrieved_well_ids: list[str], relevant_well_ids: set[str], k: int) -> float:
    """Fração dos poços relevantes que aparecem entre os top-k poços recuperados (deduplicado)."""
    if not relevant_well_ids:
        return 0.0
    top_k = set(_dedupe_by_first_occurrence(retrieved_well_ids)[:k])
    return len(top_k & relevant_well_ids) / len(relevant_well_ids)


def reciprocal_rank(retrieved_well_ids: list[str], relevant_well_ids: set[str]) -> float:
    """1/posição do primeiro poço relevante recuperado (0 se nenhum aparece), deduplicado."""
    deduped = _dedupe_by_first_occurrence(retrieved_well_ids)
    for i, well_id in enumerate(deduped, start=1):
        if well_id in relevant_well_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_well_ids: list[str], relevant_well_ids: set[str], k: int) -> float:
    """nDCG@k com relevância binária, sobre poços únicos (deduplicados por primeira ocorrência)."""
    top_k = _dedupe_by_first_occurrence(retrieved_well_ids)[:k]
    dcg = sum(
        1.0 / math.log2(i + 2) for i, well_id in enumerate(top_k) if well_id in relevant_well_ids
    )
    n_relevant_in_k = min(len(relevant_well_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_k))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalScores:
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]


def evaluate_retrieval(
    all_retrieved: list[list[str]],
    all_relevant: list[set[str]],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> RetrievalScores:
    """Agrega métricas de recuperação sobre um conjunto de perguntas.

    all_retrieved[i]: lista ordenada de well_ids recuperados para a pergunta i.
    all_relevant[i]: conjunto de well_ids considerados corretos para a pergunta i.
    Perguntas sem well_id relevante conhecido (ex.: agregadas) devem ser filtradas
    antes de chamar esta função.
    """
    n = len(all_retrieved)
    recall = {k: sum(recall_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)) / n for k in k_values}
    mrr = sum(reciprocal_rank(r, rel) for r, rel in zip(all_retrieved, all_relevant)) / n
    ndcg = {k: sum(ndcg_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)) / n for k in k_values}
    return RetrievalScores(recall_at_k=recall, mrr=mrr, ndcg_at_k=ndcg)


# --------------------------------------------------------------------------
# Métricas de resposta (avaliação da geração final)
# --------------------------------------------------------------------------

@dataclass
class AnswerJudgment:
    question_id: str
    verdict: str  # "correta", "parcial", "incorreta", "alucinacao", "nao_encontrada"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def judge_exact_field_answer(answer: str, expected: str) -> str:
    """Julgamento automático simples para perguntas de campo exato: verifica se o
    valor esperado aparece na resposta gerada (substring, normalizado).

    Não substitui avaliação humana para perguntas semânticas/abertas — serve como
    heurística de primeira passada para o subconjunto de perguntas exatas, onde a
    resposta certa é um valor curto e determinístico.
    """
    norm_answer = _normalize(answer)
    norm_expected = _normalize(expected)

    if "não foi encontrada" in norm_answer or "não encontrada" in norm_answer:
        return "nao_encontrada"
    if norm_expected in norm_answer:
        return "correta"
    # parcial: alguma palavra relevante do valor esperado aparece, mas não o valor completo
    expected_words = set(norm_expected.split())
    answer_words = set(norm_answer.split())
    if expected_words & answer_words:
        return "parcial"
    return "incorreta"


@dataclass
class AnswerEvaluationSummary:
    n_total: int
    taxa_correta: float
    taxa_parcial: float
    taxa_incorreta: float
    taxa_nao_encontrada: float
    judgments: list[AnswerJudgment]


def evaluate_exact_answers(
    question_ids: list[str], answers: list[str], expected: list[str]
) -> AnswerEvaluationSummary:
    judgments = [
        AnswerJudgment(question_id=qid, verdict=judge_exact_field_answer(ans, exp))
        for qid, ans, exp in zip(question_ids, answers, expected)
    ]
    n = len(judgments)
    counts = {"correta": 0, "parcial": 0, "incorreta": 0, "nao_encontrada": 0}
    for j in judgments:
        counts[j.verdict] = counts.get(j.verdict, 0) + 1

    return AnswerEvaluationSummary(
        n_total=n,
        taxa_correta=counts["correta"] / n if n else 0.0,
        taxa_parcial=counts["parcial"] / n if n else 0.0,
        taxa_incorreta=counts["incorreta"] / n if n else 0.0,
        taxa_nao_encontrada=counts["nao_encontrada"] / n if n else 0.0,
        judgments=judgments,
    )


def hallucination_rate(judgments: list[AnswerJudgment], is_hallucination: list[bool]) -> float:
    """Taxa de alucinação: fração de respostas marcadas como alucinação (avaliação
    manual/humana — se a resposta afirma algo com confiança que não está na evidência
    recuperada). `is_hallucination` deve ser fornecido externamente (checagem manual
    comparando resposta gerada com o texto de evidência), não é inferível automaticamente.
    """
    if not judgments:
        return 0.0
    return sum(is_hallucination) / len(is_hallucination)
