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


def build_default_registry() -> OperatorRegistry:
    from preprocessing.operators.filters.aspect_ratio import AspectRatioFilter
    from preprocessing.operators.filters.blur import PersonBlurFilter
    from preprocessing.operators.filters.compression import CompressionQualityFilter
    from preprocessing.operators.filters.duplicate import DuplicateFilter
    from preprocessing.operators.filters.framebox import FrameBoxFilter
    from preprocessing.operators.filters.gray_style import GrayStyleFilter
    from preprocessing.operators.filters.stitch import StitchLineFilterV2
    from preprocessing.operators.labelers.person_attribute import PersonAttributeLabeler
    from preprocessing.operators.labelers.object_holdable import ObjectHoldableLabeler

    registry = OperatorRegistry()
    registry.register("aspect_ratio", AspectRatioFilter)
    registry.register("gray_style", GrayStyleFilter)
    registry.register("compression_quality", CompressionQualityFilter)
    registry.register("framebox", FrameBoxFilter)
    registry.register("stitch_v2", StitchLineFilterV2)
    registry.register("person_blur", PersonBlurFilter)
    registry.register("duplicate", DuplicateFilter)
    registry.register("person_attribute_labeler", PersonAttributeLabeler)
    registry.register("object_holdable_labeler", ObjectHoldableLabeler)
    return registry


DEFAULT_REGISTRY = build_default_registry()
