from __future__ import annotations

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import OperatorResult, Sample


class FilterOperator:
    name: str

    def setup(self, context: PipelineContext) -> None:
        pass

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        raise NotImplementedError

    def teardown(self, context: PipelineContext) -> None:
        pass


class BatchFilterOperator(FilterOperator):
    def process_batch(
        self,
        samples: list[Sample],
        context: PipelineContext,
    ) -> dict[str, OperatorResult]:
        raise NotImplementedError


class LabelOperator:
    name: str

    def setup(self, context: PipelineContext) -> None:
        pass

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        raise NotImplementedError

    def teardown(self, context: PipelineContext) -> None:
        pass
