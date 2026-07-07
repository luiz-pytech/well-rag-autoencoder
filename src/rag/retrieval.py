"""Recuperação de informação: busca estruturada (metadados exatos), lexical
(TF-IDF/BM25), vetorial (embeddings) e híbrida.

Arquitetura híbrida (ver README do projeto):
1. Perguntas de campo exato (ex.: "qual a bacia do poço X?") consultam primeiro
   os metadados estruturados — resposta direta, sem ambiguidade de recuperação.
2. Perguntas semânticas/abertas consultam a base vetorial de chunks.
3. Perguntas mistas combinam as duas fontes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    chunk_id: str
    well_id: str
    text: str
    score: float
    method: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class TfidfRetriever:
    """Baseline de recuperação lexical simples (TF-IDF + similaridade de cosseno)."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(tokenizer=_tokenize, lowercase=False)
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).ravel()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunks[i]["chunk_id"],
                well_id=self.chunks[i]["well_id"],
                text=self.chunks[i]["text"],
                score=float(sims[i]),
                method="tfidf",
            )
            for i in top_idx
        ]


class BM25Retriever:
    """Baseline de recuperação lexical BM25 (mais robusto que TF-IDF puro para termos raros)."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunks[i]["chunk_id"],
                well_id=self.chunks[i]["well_id"],
                text=self.chunks[i]["text"],
                score=float(scores[i]),
                method="bm25",
            )
            for i in top_idx
        ]


class VectorRetriever:
    """Busca vetorial via embeddings (FAISS IndexFlatIP sobre vetores normalizados = cosseno)."""

    def __init__(self, chunks: list[dict], embeddings: np.ndarray):
        import faiss

        self.chunks = chunks
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        return [
            RetrievedChunk(
                chunk_id=self.chunks[i]["chunk_id"],
                well_id=self.chunks[i]["well_id"],
                text=self.chunks[i]["text"],
                score=float(scores[0][rank]),
                method="vector",
            )
            for rank, i in enumerate(indices[0])
            if i != -1
        ]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = 60, top_k: int = 5
) -> list[RetrievedChunk]:
    """Combina múltiplos rankings (ex.: lexical + vetorial) via Reciprocal Rank Fusion.

    RRF é robusto a escalas de score diferentes entre métodos (BM25 e cosseno não
    são comparáveis diretamente) — soma 1/(k + posição) de cada chunk em cada lista.
    """
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_id[chunk.chunk_id] = chunk

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk_id=chunk_by_id[cid].chunk_id,
            well_id=chunk_by_id[cid].well_id,
            text=chunk_by_id[cid].text,
            score=scores[cid],
            method="hybrid_rrf",
        )
        for cid in ranked_ids
    ]


def structured_lookup(metadata_list: list[dict], well_id: str | None = None, field: str | None = None) -> list[dict]:
    """Consulta direta aos metadados estruturados — usada para perguntas de campo exato.

    Se well_id for informado, filtra por poço. Se field for informado, retorna
    apenas esse campo (mais o well_id) em vez do registro completo.
    """
    results = metadata_list
    if well_id is not None:
        results = [m for m in results if m["well_id"].lower() == well_id.lower()]

    if field is not None:
        return [{"well_id": m["well_id"], field: m.get(field)} for m in results]
    return results
