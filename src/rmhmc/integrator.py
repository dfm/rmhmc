__all__ = ["leapfrog", "implicit_midpoint"]

from dataclasses import dataclass
from typing import Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from typing_extensions import Protocol

from .base_types import (
    KineticFunction,
    KineticState,
    Momentum,
    Position,
    PotentialFunction,
    Scalar,
)


@dataclass(frozen=True)
class IntegratorState:
    q: Position
    p: Momentum


@dataclass(frozen=True)
class LeapfrogState(IntegratorState):
    dUdq: Position


class LeapfrogInitFunction(Protocol):
    def __call__(
        self, __kinetic_state: KineticState, __q: Position, __p: Momentum
    ) -> LeapfrogState:
        ...


class LeapfrogUpdateFunction(Protocol):
    def __call__(
        self,
        __kinetic_state: KineticState,
        __state: LeapfrogState,
        *,
        epsilon: Scalar
    ) -> Tuple[LeapfrogState, bool]:
        ...


def leapfrog(
    potential_fn: PotentialFunction, kinetic_fn: KineticFunction
) -> Tuple[LeapfrogInitFunction, LeapfrogUpdateFunction]:
    dU = jax.grad(potential_fn)

    def init_fn(_: KineticState, q: Position, p: Momentum) -> LeapfrogState:
        return LeapfrogState(q, p, dU(q))

    def update_fn(
        kinetic_state: KineticState, state: LeapfrogState, *, epsilon: Scalar
    ) -> Tuple[LeapfrogState, bool]:
        p = tree_map(
            lambda p, dUdq: p - 0.5 * epsilon * dUdq, state.p, state.dUdq
        )
        dTdp = jax.grad(kinetic_fn, argnums=2)(kinetic_state, None, p)
        q = tree_map(lambda q, dTdp: q + epsilon * dTdp, state.q, dTdp)
        dUdq = dU(q)
        p = tree_map(lambda p, dUdq: p - 0.5 * epsilon * dUdq, p, dUdq)
        return LeapfrogState(q, p, dUdq), True

    return init_fn, update_fn


@dataclass(frozen=True)
class ImplicitMidpointState(IntegratorState):
    dHdq: Position
    dHdp: Momentum


class ImplicitMidpointInitFunction(Protocol):
    def __call__(
        self, __kinetic_state: KineticState, __q: Position, __p: Momentum
    ) -> ImplicitMidpointState:
        ...


class ImplicitMidpointUpdateFunction(Protocol):
    def __call__(
        self,
        __kinetic_state: KineticState,
        __state: ImplicitMidpointState,
        *,
        epsilon: Scalar
    ) -> Tuple[ImplicitMidpointState, bool]:
        ...


def implicit_midpoint(
    potential_fn: PotentialFunction,
    kinetic_fn: KineticFunction,
    **solver_kwargs: Any
) -> Tuple[ImplicitMidpointInitFunction, ImplicitMidpointUpdateFunction]:
    hamiltonian = lambda state, q, p: potential_fn(q) + kinetic_fn(state, q, p)
    vector_field = jax.grad(hamiltonian, argnums=(1, 2))

    def init_fn(
        kinetic_state: KineticState, q: Position, p: Momentum
    ) -> ImplicitMidpointState:
        dHdq, dHdp = vector_field(kinetic_state, q, p)
        return ImplicitMidpointState(q, p, dHdq, dHdp)

    def update_fn(
        kinetic_state: KineticState,
        state: ImplicitMidpointState,
        *,
        epsilon: Scalar
    ) -> Tuple[ImplicitMidpointState, bool]:
        def step(args: Tuple[Position, Momentum]) -> Tuple[Position, Momentum]:
            q, p = args
            dHdq, dHdp = vector_field(kinetic_state, q, p)
            return (
                state.q + 0.5 * epsilon * dHdp,
                state.p - 0.5 * epsilon * dHdq,
            )

        # Use an initial half step using the pre-computed vector field
        q = state.q + 0.5 * epsilon * state.dHdp
        p = state.p - 0.5 * epsilon * state.dHdq

        # Solve for the midpoint
        (q, p), success = solve_fixed_point(step, (q, p), **solver_kwargs)

        # Compute the resulting vector field and update the state
        dHdq = (2.0 / epsilon) * (state.p - p)
        dHdp = (2.0 / epsilon) * (q - state.q)
        q = 2 * q - state.q
        p = 2 * p - state.p

        return ImplicitMidpointState(q, p, dHdq, dHdp), success

    return init_fn, update_fn


def solve_fixed_point(
    func: Callable[[Any], Any],
    x0: Any,
    convergence_tol: float = 1e-6,
    divergence_tol: float = 1e10,
    max_iters: int = 100,
    norm_fn: Callable[[Any], Scalar] = lambda x: jnp.max(jnp.abs(x)),
) -> Tuple[Any, bool]:
    def compute_norm(x: Any, xp: Any) -> Scalar:
        return norm_fn(ravel_pytree(tree_map(jnp.subtract, x, xp))[0])

    def cond_fn(args: Tuple[int, Any, Any]) -> bool:
        n, _, norm = args
        return (
            (n < max_iters)
            & jnp.isfinite(norm)
            & (norm < divergence_tol)
            & (norm > convergence_tol)
        )

    def body_fn(args: Tuple[int, Any, Any]) -> Tuple[int, Any, Any]:
        n, x, _ = args
        xn = func(x)
        norm = compute_norm(xn, x)
        return n + 1, xn, norm

    x = func(x0)
    n, x, norm = jax.lax.while_loop(
        cond_fn, body_fn, (0, x, compute_norm(x, x0))
    )
    success = jnp.isfinite(norm) & (norm <= convergence_tol)
    return x, success


IntegratorInitFunction = Union[
    LeapfrogInitFunction, ImplicitMidpointInitFunction
]
IntegratorUpdateFunction = Union[
    LeapfrogUpdateFunction, ImplicitMidpointUpdateFunction
]
