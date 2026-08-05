"""Dependency-free SentenceTransformer fake used by the worker protocol test."""

from __future__ import annotations

import os
from pathlib import Path


def _record(value: str) -> None:
    counter_path = os.environ.get("ROOMIE_FAKE_SENTENCE_TRANSFORMER_COUNTER", "")
    if counter_path:
        with Path(counter_path).open("a", encoding="utf-8") as stream:
            stream.write(value + "\n")


class SentenceTransformer:
    def __init__(self, model: str, device: str) -> None:
        del model, device
        _record("load")

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, documents, **kwargs):
        del kwargs
        _record("encode:" + str(len(documents)))
        if "__raise__" in documents:
            raise RuntimeError("synthetic SentenceTransformer error")
        return [
            [float(len(document) + 1), 2.0, 3.0]
            for document in documents
        ]
