__all__ = [
    "euclidean",
    "riemannian",
    "compute_total_energy",
    "integrate",
    "integrate_trajectory",
]

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax, random
from jax.flatten_util import ravel_pytree

from .base_types import (
    Array,
    EuclideanKineticState,
    KineticFunction,
    KineticState,
    Momentum,
    Position,
    PotentialFunction,
    Scalar,
)
from .integrator import (
    IntegratorInitFunction,
    IntegratorState,
    IntegratorUpdateFunction,
    implicit_midpoint,
    leapfrog,
)


class System(NamedTuple):
    # Functions to evaluate the potential and kinetic energies
    potential: PotentialFunction
    kinetic: KineticFunction

    # Function to sample the momentum at a specific state and position
    sample_momentum: Callable[
        [KineticState, Position, random.KeyArray], Momentum
    ]

    # Functions for tuning the kinetic energy function
    kinetic_tune_init: Callable[[int], KineticState]
    kinetic_tune_update: Callable[[KineticState, Position], KineticState]
    kinetic_tune_finish: Callable[[KineticState], KineticState]

    # System-specific integrator functions
    integrator_init: IntegratorInitFunction
    integrator_update: IntegratorUpdateFunction


def euclidean(
    log_probability_fn: Callable[[Position], Scalar],
    cov: Optional[Array] = None,
    tril: Optional[Array] = None,
    diagonal: bool = True,
) -> System:
    def potential(q: Position) -> Scalar:
        return -log_probability_fn(q)

    def kinetic(state: KineticState, _: Position, p: Momentum) -> Scalar:
        assert isinstance(state, EuclideanKineticState)
        p, _ = ravel_pytree(p)
        tril = state.tril
        if tril.ndim == 1:
            alpha = tril * p
        else:
            alpha = jnp.dot(tril.T, p)
        return 0.5 * jnp.sum(jnp.square(alpha))

    def sample_momentum(
        state: KineticState, q: Position, rng_key: random.KeyArray
    ) -> Momentum:
        assert isinstance(state, EuclideanKineticState)
        q, unravel = ravel_pytree(q)
        tril = state.tril
        eps = random.normal(rng_key, q.shape)
        if tril.ndim == 1:
            return unravel(eps / tril)
        return unravel(
            jsp.linalg.solve_triangular(tril, eps, trans=1, lower=True)
        )

    def kinetic_tune_init(size: int) -> KineticState:
        if cov is not None:
            assert tril is None
            if cov.ndim == 2:
                tril_ = jsp.linalg.cholesky(cov, lower=True)
            else:
                tril_ = jnp.sqrt(cov)
            shape = tril_.shape
            assert shape == (size, size) or shape == (size,)
        elif tril is not None:
            tril_ = tril
            shape = tril_.shape
            assert shape == (size, size) or shape == (size,)
        else:
            shape = (size,) if diagonal else (size, size)
            tril_ = jnp.ones(size) if diagonal else jnp.eye(size)
        mu = jnp.zeros(shape[-1])
        m2 = jnp.zeros(shape)
        return EuclideanKineticState(count=0, tril=tril_, mu=mu, m2=m2)

    def kinetic_tune_update(state: KineticState, q: Position) -> KineticState:
        assert isinstance(state, EuclideanKineticState)
        q, _ = ravel_pytree(q)
        n = state.count + 1
        d1 = q - state.mu
        mu = state.mu + d1 / n
        d2 = q - mu
        if state.m2.ndim == 1:
            m2 = state.m2 + d1 * d2
        else:
            m2 = state.m2 + jnp.outer(d2, d1)
        return EuclideanKineticState(count=n, tril=state.tril, mu=mu, m2=m2)

    def kinetic_tune_finish(state: KineticState) -> KineticState:
        assert isinstance(state, EuclideanKineticState)
        cov = state.m2 / (state.count - 1)
        if state.m2.ndim == 2:
            tril = jsp.linalg.cholesky(cov, lower=True)
        else:
            tril = jnp.sqrt(cov)
        return EuclideanKineticState(
            count=0,
            tril=tril,
            mu=jnp.zeros_like(state.mu),
            m2=jnp.zeros_like(state.m2),
        )

    integrator_init, integrator_update = leapfrog(potential, kinetic)

    return System(
        potential=potential,
        kinetic=kinetic,
        sample_momentum=sample_momentum,
        kinetic_tune_init=kinetic_tune_init,
        kinetic_tune_update=kinetic_tune_update,
        kinetic_tune_finish=kinetic_tune_finish,
        integrator_init=integrator_init,
        integrator_update=integrator_update,
    )


def riemannian(
    log_probability_fn: Callable[[Position], Scalar],
    metric_fn: Callable[[Position], Array],
) -> System:
    def potential(q: Position) -> Scalar:
        return -log_probability_fn(q)

    def kinetic(_: KineticState, q: Position, p: Momentum) -> Scalar:
        p, _ = ravel_pytree(p)
        metric = metric_fn(q)
        tril, _ = jsp.linalg.cho_factor(metric, lower=True)
        half_log_det = jnp.sum(jnp.log(jnp.diag(tril)))
        alpha = jsp.linalg.solve_triangular(tril, p, lower=True)
        return 0.5 * jnp.sum(jnp.square(alpha)) + p.size * half_log_det

    def sample_momentum(
        _: KineticState, q: Position, rng_key: random.KeyArray
    ) -> Momentum:
        metric = metric_fn(q)
        q, unravel = ravel_pytree(q)
        tril, _ = jsp.linalg.cho_factor(metric, lower=True)
        eps = random.normal(rng_key, q.shape)
        return unravel(jnp.dot(tril, eps))

    # This metric doesn't have any tuning parameters
    def kinetic_tune_init(_: int) -> KineticState:
        return None

    def kinetic_tune_update(state: KineticState, q: Position) -> KineticState:
        return None

    def kinetic_tune_finish(state: KineticState) -> KineticState:
        return None

    integrator_init, integrator_update = implicit_midpoint(potential, kinetic)

    return System(
        potential=potential,
        kinetic=kinetic,
        sample_momentum=sample_momentum,
        kinetic_tune_init=kinetic_tune_init,
        kinetic_tune_update=kinetic_tune_update,
        kinetic_tune_finish=kinetic_tune_finish,
        integrator_init=integrator_init,
        integrator_update=integrator_update,
    )


def compute_total_energy(
    system: System, kinetic_state: KineticState, q: Position, p: Momentum
) -> Scalar:
    return system.potential(q) + system.kinetic(kinetic_state, q, p)


def integrate(
    system: System,
    num_steps: Scalar,
    step_size: Scalar,
    kinetic_state: KineticState,
    state: IntegratorState,
) -> Tuple[IntegratorState, bool]:
    def step(
        _: Scalar, carry: Tuple[IntegratorState, bool]
    ) -> Tuple[IntegratorState, bool]:
        state_, success_ = system.integrator_update(
            step_size, kinetic_state, carry[0]
        )
        return state_, jnp.logical_and(success_, carry[1])

    return lax.fori_loop(0, num_steps, step, (state, True))


def integrate_trajectory(
    system: System,
    num_steps: int,
    step_size: Scalar,
    kinetic_state: KineticState,
    state: IntegratorState,
) -> Tuple[IntegratorState, Array]:
    def step(
        carry: Tuple[IntegratorState, bool], _: Any
    ) -> Tuple[Tuple[IntegratorState, bool], Tuple[IntegratorState, bool]]:
        state_, success_ = system.integrator_update(
            step_size, kinetic_state, carry[0]
        )
        result = state_, jnp.logical_and(success_, carry[1])
        return result, result

    return lax.scan(step, (state, True), jnp.arange(num_steps))[1]
