from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import numpy as np
import imagehash
from tqdm import tqdm

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import BatchFilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


@dataclass
class ImageInfo:
    sample: Sample
    phash: imagehash.ImageHash
    hash_int: int
    size: int


class DuplicateFilter(BatchFilterOperator):
    name = "duplicate"

    def __init__(self, threshold: int = 9, hash_size: int = 8, window_size: int = 50) -> None:
        self.threshold = int(threshold)
        self.hash_size = int(hash_size)
        self.window_size = int(window_size)

    def process_batch(
        self,
        samples: list[Sample],
        context: PipelineContext,
    ) -> dict[str, OperatorResult]:
        results: dict[str, OperatorResult] = {}
        infos: list[ImageInfo] = []

        for sample in tqdm(samples, desc="Dedupe:phash", unit="img"):
            try:
                phash = self._compute_phash(sample.source_path)
                infos.append(ImageInfo(sample=sample, phash=phash, hash_int=self._phash_to_int(phash), size=sample.source_path.stat().st_size))
            except Exception as exc:
                results[sample.sample_id] = OperatorResult.error(f"hash_failed: {exc!r}")

        infos.sort(key=lambda info: info.hash_int)

        kept: list[ImageInfo] = []
        for i, current in enumerate(tqdm(infos, desc="Dedupe:compare", unit="img")):
            duplicate_of: ImageInfo | None = None
            distance = 0
            window_start = max(0, i - self.window_size)
            window_end = min(len(infos), i + self.window_size + 1)
            for j in range(window_start, window_end):
                if j == i:
                    continue
                existing = infos[j]
                if existing not in kept:
                    continue
                distance = current.phash - existing.phash
                if distance <= self.threshold:
                    duplicate_of = existing
                    break

            if duplicate_of is None:
                kept.append(current)
                results[current.sample.sample_id] = OperatorResult.pass_({"phash": str(current.phash)})
                continue

            keeper, duplicate = self._choose_keeper(duplicate_of, current)
            if keeper.sample.sample_id != duplicate_of.sample.sample_id:
                kept.remove(duplicate_of)
                kept.append(keeper)
                results[keeper.sample.sample_id] = OperatorResult.pass_({"phash": str(keeper.phash)})

            results[duplicate.sample.sample_id] = OperatorResult.reject(
                "duplicate_image",
                {
                    "keeper": keeper.sample.sample_id,
                    "hamming_distance": distance,
                    "threshold": self.threshold,
                },
            )

        return results

    def _compute_phash(self, path: Path) -> imagehash.ImageHash:
        with Image.open(path) as image:
            return imagehash.phash(image.convert("RGB"), hash_size=self.hash_size)

    @staticmethod
    def _phash_to_int(phash: imagehash.ImageHash) -> int:
        return int.from_bytes(np.packbits(phash.hash.flatten()).tobytes(), byteorder="big")

    def _choose_keeper(self, a: ImageInfo, b: ImageInfo) -> tuple[ImageInfo, ImageInfo]:
        if a.size != b.size:
            return (a, b) if a.size > b.size else (b, a)
        return (a, b) if len(str(a.sample.source_path)) <= len(str(b.sample.source_path)) else (b, a)
