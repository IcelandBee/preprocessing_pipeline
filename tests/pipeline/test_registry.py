import pytest

from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.registry import OperatorRegistry


class DummyFilter(FilterOperator):
    name = "dummy"


def test_registry_creates_operator_by_name():
    registry = OperatorRegistry()
    registry.register("dummy", DummyFilter)

    operator = registry.create("dummy", {})

    assert isinstance(operator, DummyFilter)


def test_registry_rejects_unknown_operator():
    registry = OperatorRegistry()

    with pytest.raises(KeyError, match="unknown"):
        registry.create("unknown", {})
