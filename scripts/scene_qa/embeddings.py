"""Object search backends for Roomie scene QA."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .graph_store import GraphStore, ObjectRecord


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class SearchResult:
    record: ObjectRecord
    score: float


def _tokenize(text: str) -> set[str]:
    return {tok.lower().replace("_", " ") for tok in TOKEN_RE.findall(text)}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1.0e-12)


class ObjectSearchIndex:
    """Semantic search over object label and description text."""

    def __init__(
        self,
        graph: GraphStore,
        *,
        model_path: str | Path,
        backend: str = "embedding",
        device: str = "auto",
    ):
        if backend not in {"embedding", "lexical"}:
            raise ValueError("backend must be 'embedding' or 'lexical'")
        self.graph = graph
        self.model_path = Path(model_path).expanduser()
        self.backend = backend
        self.device = device
        self._model = None
        self._records: list[ObjectRecord] | None = None
        self._texts: list[str] | None = None
        self._embeddings: np.ndarray | None = None
        self._token_sets: list[set[str]] | None = None

    def search(
        self,
        description: str,
        *,
        top_k: int,
        room_id: int | None = None,
    ) -> list[SearchResult]:
        if not description.strip():
            return []
        top_k = max(1, min(int(top_k), 100))
        if self.backend == "lexical":
            return self._lexical_search(description, top_k=top_k, room_id=room_id)
        return self._embedding_search(description, top_k=top_k, room_id=room_id)

    def _records_and_texts(self) -> tuple[list[ObjectRecord], list[str]]:
        if self._records is None or self._texts is None:
            records = self.graph.object_records()
            self._records = records
            self._texts = [self.graph.object_text(record) for record in records]
        return self._records, self._texts

    def _embedding_search(
        self,
        description: str,
        *,
        top_k: int,
        room_id: int | None,
    ) -> list[SearchResult]:
        records, _ = self._records_and_texts()
        embeddings = self._get_object_embeddings()
        query = self._encode([description])
        scores = embeddings @ query[0]
        candidate_indices = self._candidate_indices(records, room_id)
        ranked = sorted(candidate_indices, key=lambda idx: float(scores[idx]), reverse=True)
        return [
            SearchResult(records[idx], float(scores[idx]))
            for idx in ranked[:top_k]
        ]

    def _lexical_search(
        self,
        description: str,
        *,
        top_k: int,
        room_id: int | None,
    ) -> list[SearchResult]:
        records, texts = self._records_and_texts()
        if self._token_sets is None:
            self._token_sets = [_tokenize(text) for text in texts]
        query_tokens = _tokenize(description)
        query_lower = description.lower()
        scored: list[SearchResult] = []
        for idx in self._candidate_indices(records, room_id):
            record = records[idx]
            text_lower = texts[idx].lower()
            tokens = self._token_sets[idx]
            overlap = len(tokens & query_tokens)
            denom = math.sqrt(max(1, len(tokens)) * max(1, len(query_tokens)))
            score = overlap / denom
            if record.label.lower().replace("_", " ") in query_lower:
                score += 0.35
            if query_lower in text_lower:
                score += 0.25
            if record.quality is not None:
                score += 0.01 * record.quality
            scored.append(SearchResult(record, score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _candidate_indices(
        self,
        records: Iterable[ObjectRecord],
        room_id: int | None,
    ) -> list[int]:
        indices = []
        for idx, record in enumerate(records):
            if room_id is not None and int(room_id) not in record.room_ids:
                continue
            indices.append(idx)
        return indices

    def _get_object_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            _, texts = self._records_and_texts()
            if not texts:
                self._embeddings = np.zeros((0, 1), dtype=np.float32)
            else:
                self._embeddings = self._encode(texts)
        return self._embeddings

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return _normalize_rows(arr)

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embedding search. "
                "Use --embedding-backend lexical for dependency-light tests."
            ) from exc

        device = self._resolve_device()
        model_path = self.model_path.expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"embedding model path does not exist: {model_path}")
        self._model = SentenceTransformer(str(model_path), device=device)
        return self._model

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
