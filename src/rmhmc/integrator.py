__all__ = ["leapfrog", "implicit_midpoint"]

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from .base_types import (
    Array,
    KineticFunction,
    KineticState,
    Momentum,
    Position,
    PotentialFunction,
    Scalar,
)


class LeapfrogState(NamedTuple):
    q: Position
    p: Momentum
    dUdq: Position


class ImplicitMidpointState(NamedTuple):
    q: Position
    p: Momentum
    dHdq: Position
    dHdp: Momentum


IntegratorState = Union[LeapfrogState, ImplicitMidpointState]
IntegratorInitFunction = Callable[
    [KineticState, Position, Momentum], IntegratorState
]
IntegratorUpdateFunction = Callable[
    [Scalar, KineticState, IntegratorState], Tuple[IntegratorState, bool]
]


def _axpby(a: Scalar, x: Array, y: Array, b: Optional[Scalar] = None) -> Array:
    if b is None:
        return tree_map(lambda x_, y_: a * x_ + y_, x, y)
    return tree_map(lambda x_, y_: a * x_ + b * y_, x, y)


def leapfrog(
    potential_fn: PotentialFunction, kinetic_fn: KineticFunction
) -> Tuple[IntegratorInitFunction, IntegratorUpdateFunction]:
    dU = jax.grad(potential_fn)

    def init_fn(_: KineticState, q: Position, p: Momentum) -> IntegratorState:
        return LeapfrogState(q, p, dU(q))

    def update_fn(
        step_size: Scalar, kinetic_state: KineticState, state: IntegratorState
    ) -> Tuple[IntegratorState, bool]:
        assert isinstance(state, LeapfrogState)
        p = _axpby(-0.5 * step_size, state.dUdq, state.p)
        dTdp = jax.grad(kinetic_fn, argnums=2)(kinetic_state, None, p)
        q = _axpby(step_size, dTdp, state.q)
        dUdq = dU(q)
        p = _axpby(-0.5 * step_size, dUdq, p)
        return LeapfrogState(q, p, dUdq), True

    return init_fn, update_fn


def implicit_midpoint(
    potential_fn: PotentialFunction,
    kinetic_fn: KineticFunction,
    **solver_kwargs: Any
) -> Tuple[IntegratorInitFunction, IntegratorUpdateFunction]:
    hamiltonian = lambda state, q, p: potential_fn(q) + kinetic_fn(state, q, p)
    vector_field = jax.grad(hamiltonian, argnums=(1, 2))

    def init_fn(
        kinetic_state: KineticState, q: Position, p: Momentum
    ) -> IntegratorState:
        dHdq, dHdp = vector_field(kinetic_state, q, p)
        return ImplicitMidpointState(q, p, dHdq, dHdp)

    def update_fn(
        step_size: Scalar,
        kinetic_state: KineticState,
        state: IntegratorState,
    ) -> Tuple[IntegratorState, bool]:
        assert isinstance(state, ImplicitMidpointState)

        def step(args: Tuple[Position, Momentum]) -> Tuple[Position, Momentum]:
            q, p = args
            dHdq, dHdp = vector_field(kinetic_state, q, p)
            return (
                _axpby(0.5 * step_size, dHdp, state.q),
                _axpby(-0.5 * step_size, dHdq, state.p),
            )

        # Use an initial half step using the pre-computed vector field
        q = _axpby(0.5 * step_size, state.dHdp, state.q)
        p = _axpby(-0.5 * step_size, state.dHdq, state.p)

        # Solve for the midpoint
        (q, p), success = solve_fixed_point(step, (q, p), **solver_kwargs)

        # Compute the resulting vector field and update the state
        a = 2.0 / step_size
        dHdq = _axpby(-a, p, state.p, a)
        dHdp = _axpby(a, q, state.q, -a)
        q = _axpby(2.0, q, state.q, -1.0)
        p = _axpby(2.0, p, state.p, -1.0)

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
