"""Geração de embeddings para os chunks textuais, usando um modelo multilíngue local."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"

# O modelo e5 espera prefixos específicos por convenção de treino:
# "query: " para textos de consulta, "passage: " para textos indexados/documentos.
# Omitir o prefixo degrada a qualidade do embedding nesse modelo.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_model_cache: dict[str, object] = {}


def _get_model(model_name: str = MODEL_NAME):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_passages(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Gera embeddings para textos a serem indexados (chunks/documentos)."""
    model = _get_model(model_name)
    prefixed = [PASSAGE_PREFIX + t for t in texts]
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)


def embed_queries(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Gera embeddings para textos de consulta (perguntas do usuário)."""
    model = _get_model(model_name)
    prefixed = [QUERY_PREFIX + t for t in texts]
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)


def save_embeddings(embeddings: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)


def load_embeddings(path: str | Path) -> np.ndarray:
    return np.load(path)
