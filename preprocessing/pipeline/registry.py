from __future__ import annotations

from preprocessing.pipeline.operators import BatchFilterOperator, FilterOperator, LabelOperator

OperatorInstance = FilterOperator | BatchFilterOperator | LabelOperator
OperatorClass = type[FilterOperator] | type[BatchFilterOperator] | type[LabelOperator]


class OperatorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, OperatorClass] = {}

    def register(self, name: str, factory: OperatorClass) -> None:
        self._factories[name] = factory

    def create(self, name: str, params: dict[str, object]) -> OperatorInstance:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown operator: {name}. Available operators: {available}") from exc
        return factory(**params)


DEFAULT_REGISTRY = OperatorRegistry()
