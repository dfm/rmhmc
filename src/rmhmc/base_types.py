from typing import Any, Callable, NamedTuple, Optional

from jax import random


def identity(x: Any) -> Any:
    return x


Array = Any
Position = Any
Momentum = Any
Scalar = Any


class DualAveragingState(NamedTuple):
    count: Scalar
    x: Scalar
    x_avg: Scalar
    g_avg: Scalar
    mu: Scalar


class EuclideanKineticState(NamedTuple):
    count: Scalar
    tril: Array
    mu: Array
    m2: Array


KineticState = Optional[EuclideanKineticState]
PotentialFunction = Callable[[Position], Scalar]
KineticFunction = Callable[[KineticState, Position, Momentum], Scalar]


class SamplerTuningState(NamedTuple):
    step_size: Scalar
    step_size_state: DualAveragingState
    kinetic_state: KineticState


class ProposalStats(NamedTuple):
    accept_prob: Scalar
    accept: Scalar
    diverging: Scalar


class HMCState(NamedTuple):
    q: Position
    potential_energy: Scalar
    potential_energy_grad: Position


SamplerState = HMCState


class SamplerCarry(NamedTuple):
    tuning: SamplerTuningState
    state: SamplerState
    stats: ProposalStats


class MCMCKernel(NamedTuple):
    init: Callable[[Position], SamplerCarry]
    step: Callable[[SamplerCarry, random.KeyArray], SamplerCarry]
    reset: Callable[[SamplerCarry], SamplerCarry]
    fast_update: Callable[[SamplerCarry], SamplerCarry]
    slow_update: Callable[[SamplerCarry], SamplerCarry]
    tune_finish: Callable[[SamplerCarry], SamplerCarry]
