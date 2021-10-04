__all__ = [""]

from collections import namedtuple
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from rmhmc.base_types import KineticState, Position, Scalar
from rmhmc.hamiltonian import System, integrate
from rmhmc.integrator import LeapfrogState
from rmhmc.step_size import DualAveragingState, dual_averaging


class HMCState(NamedTuple):
    q: Position
    potential_energy: Scalar
    potential_energy_grad: Position
    accept_prob: Scalar
    accept: Scalar
    diverging: Scalar
    rng_key: random.KeyArray
    kinetic_state: KineticState
    step_size_state: DualAveragingState


def hmc(
    system: System,
    rng_key: random.KeyArray,
    num_steps: int = 50,
    max_delta_energy: float = 1000.0,
    initial_step_size: float = 1.0,
) -> Tuple[
    Callable[[Position], HMCState], Callable[[bool, HMCState], HMCState]
]:
    step_size_init_fn, step_size_update_fn = dual_averaging()
    potential_and_grad = jax.value_and_grad(system.potential)

    def init_fn(q: Position) -> HMCState:
        U, dU = potential_and_grad(q)
        return HMCState(
            q=q,
            potential_energy=U,
            potential_energy_grad=dU,
            accept_prob=0.0,
            accept=False,
            diverging=False,
            rng_key=rng_key,
            kinetic_state=system.kinetic_tune_init(ravel_pytree(q)[0].size),
            step_size_state=step_size_init_fn(jnp.log(initial_step_size)),
        )

    def update_fn(tuning: bool, state: HMCState) -> HMCState:
        rng_mom, rng_accept, rng_next = random.split(state.rng_key, 3)

        p = system.sample_momentum(state.kinetic_state, state.q, rng_mom)
        initial_coords = system.integrator_init(
            state.kinetic_state, state.q, p
        )
        initial_energy = state.potential_energy + system.kinetic(
            state.kinetic_state, state.q, p
        )

        if tuning:
            step_size = jnp.exp(state.step_size_state.x)
        else:
            step_size = jnp.exp(state.step_size_state.x_avg)

        final_coords, success = integrate(
            system, num_steps, step_size, state.kinetic_state, initial_coords
        )

        if isinstance(final_coords, LeapfrogState):
            potential_energy = system.potential(final_coords.q)
            potential_energy_grad = final_coords.dUdq
        else:
            potential_energy, potential_energy_grad = potential_and_grad(
                final_coords.q
            )
        kinetic_energy = system.kinetic(
            state.kinetic_state, final_coords.q, final_coords.p
        )
        energy = potential_energy + kinetic_energy

        delta = energy - initial_energy
        delta = jnp.where(jnp.isnan(delta), jnp.inf, delta)
        diverging = jnp.logical_not(success) | (delta > max_delta_energy)
        accept_prob = jnp.clip(jnp.exp(-delta), a_max=1.0)
        accept = random.bernoulli(rng_accept, accept_prob)

        step_size_state = state.step_size_state
        if tuning:
            step_size_state = step_size_update_fn(accept_prob, step_size_state)

        new_state = HMCState(
            q=final_coords.q,
            potential_energy=potential_energy,
            potential_energy_grad=potential_energy_grad,
            accept_prob=accept_prob,
            accept=accept,
            diverging=diverging,
            rng_key=rng_next,
            kinetic_state=state.kinetic_state,
            step_size_state=step_size_state,
        )

        return jax.lax.cond(
            accept, lambda _: new_state, lambda _: state, operand=None
        )

    return init_fn, update_fn
