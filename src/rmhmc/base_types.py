from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

from typing_extensions import Protocol

Array = Any
Position = Any
Momentum = Any
Scalar = Any


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
