import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
from typing_extensions import Protocol


def register_pytree_node_dataclass(cls: Any) -> Any:
    _flatten = lambda obj: jax.tree_flatten(dataclasses.asdict(obj))
    _unflatten = lambda d, children: cls(**d.unflatten(children))
    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls


Array = Any
Position = Any
Momentum = Any
Scalar = Any


@register_pytree_node_dataclass
@dataclass(frozen=True)
class KineticState:
    pass


class PotentialFunction(Protocol):
    def __call__(self, __q: Position) -> Scalar:
        ...


class KineticFunction(Protocol):
    def __call__(
        self, __kinetic_state: KineticState, __q: Position, __p: Momentum
    ) -> Scalar:
        ...
