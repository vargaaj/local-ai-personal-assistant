from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .cuda import (
    active_onnx_providers,
    assert_cuda_provider_available,
    prepare_cuda_runtime,
)
from .tracing import timed, trace


class EmbeddingProvider(Protocol):
    model_name: str
    vector_size: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class FastEmbedProvider:
    def __init__(
        self,
        model_name: str,
        vector_size: int,
        *,
        cuda: bool,
        providers: list[str],
        device_ids: list[int] | None,
        batch_size: int,
    ) -> None:
        self.model_name = model_name
        self.vector_size = vector_size
        self.cuda = cuda
        self.providers = providers
        self.device_ids = device_ids
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.cuda:
                prepare_cuda_runtime()

            from fastembed import TextEmbedding

            if self.cuda:
                assert_cuda_provider_available()

            cuda_arg = False if self.providers else self.cuda
            self._model = TextEmbedding(
                model_name=self.model_name,
                providers=self.providers or None,
                cuda=cuda_arg,
                device_ids=self.device_ids,
            )
            if self.cuda:
                self._assert_cuda_provider_active()
        return self._model

    @trace
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        vectors = []
        with timed(
            "embeddings.embed_texts",
            model=self.model_name,
            count=len(texts),
            cuda=self.cuda,
        ):
            for vector in self.model.embed(texts, batch_size=self.batch_size):
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                vectors.append([float(value) for value in vector])
        return vectors

    @trace
    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []

    def _assert_cuda_provider_active(self) -> None:
        active = set(active_onnx_providers(self._model))
        if active and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                "FastEmbed loaded without CUDAExecutionProvider even though GPU "
                f"embeddings are enabled. Active providers: {sorted(active)}."
            )
