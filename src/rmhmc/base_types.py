from typing import Any, Callable, NamedTuple, Optional

Array = Any
Position = Any
Momentum = Any
Scalar = Any


class EuclideanKineticState(NamedTuple):
    count: Scalar
    tril: Array
    mu: Array
    m2: Array


KineticState = Optional[EuclideanKineticState]
PotentialFunction = Callable[[Position], Scalar]
KineticFunction = Callable[[KineticState, Position, Momentum], Scalar]
