"""PatchCore with a lower peak GPU memory during fitting. Same maths.

anomalib collects each batch's patch embeddings in a list, then stacks the
list into one tensor. For a moment both the list and the stacked copy are on
the GPU: for hazelnut (391 images, 400k patches, 1536 numbers each) that is
about 4.9 GB, which overflows a 6 GB card into slow shared memory.

This subclass writes each batch straight into one pre-sized tensor instead, so
there is only ever one copy. Embedding order, random projection and coreset
selection are untouched, so the scores come out identical (checked on bottle).
"""

from __future__ import annotations

import torch
from anomalib.data import Batch
from anomalib.models import Patchcore
from anomalib.models.components.sampling import KCenterGreedy


class LeanPatchcore(Patchcore):
    _bank: torch.Tensor | None = None
    _filled: int = 0

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        self.model(batch.image)
        embedding = self.model.embedding_store.pop()  # this batch's patches, (patches, dims)
        if self._bank is None:
            patches_per_image = embedding.shape[0] // batch.image.shape[0]
            total = len(self.trainer.datamodule.train_data) * patches_per_image
            self._bank = torch.empty((total, embedding.shape[1]), dtype=embedding.dtype, device=embedding.device)
            self._filled = 0
        self._bank[self._filled : self._filled + embedding.shape[0]] = embedding
        self._filled += embedding.shape[0]
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        if self._bank is None or self._filled != self._bank.shape[0]:
            msg = f"Expected a full embedding buffer, got {self._filled} rows."
            raise ValueError(msg)
        sampler = KCenterGreedy(embedding=self._bank, sampling_ratio=self.coreset_sampling_ratio)
        self.model.memory_bank = sampler.sample_coreset()
        self._bank = None
        self._filled = 0
