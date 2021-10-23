__all__ = ["hmc"]

from typing import Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from rmhmc.base_types import (
    HMCState,
    MCMCKernel,
    Position,
    ProposalStats,
    SamplerCarry,
    SamplerTuningState,
    Scalar,
)
from rmhmc.hamiltonian import System, integrate
from rmhmc.integrator import LeapfrogState
from rmhmc.step_size import dual_averaging


def hmc(
    system: System,
    num_steps: Union[int, Callable[[random.KeyArray], Scalar]] = 50,
    max_delta_energy: float = 1000.0,
    initial_step_size: float = 1.0,
    target_accept_prob: float = 0.8,
    dual_averaging_args: Optional[Dict[str, float]] = None,
) -> MCMCKernel:
    step_size_init_fn, step_size_update_fn = dual_averaging(
        **({} if dual_averaging_args is None else dual_averaging_args)
    )
    potential_and_grad = jax.value_and_grad(system.potential)

    def init_fn(q: Position) -> SamplerCarry:
        U, dU = potential_and_grad(q)
        return SamplerCarry(
            tuning=SamplerTuningState(
                step_size=initial_step_size,
                step_size_state=step_size_init_fn(
                    jnp.log(10.0 * initial_step_size)
                ),
                kinetic_state=system.kinetic_tune_init(
                    ravel_pytree(q)[0].size
                ),
            ),
            state=HMCState(
                q=q,
                potential_energy=U,
                potential_energy_grad=dU,
            ),
            stats=ProposalStats(
                accept_prob=1.0,
                accept=True,
                diverging=False,
            ),
        )

    def step_fn(carry: SamplerCarry, rng_key: random.KeyArray) -> SamplerCarry:
        tuning_state, state, _ = carry

        if callable(num_steps):
            rng_steps, rng_mom, rng_accept = random.split(rng_key, 3)
            actual_num_steps = num_steps(rng_steps)
        else:
            rng_mom, rng_accept = random.split(rng_key)
            actual_num_steps = num_steps

        p = system.sample_momentum(
            tuning_state.kinetic_state, state.q, rng_mom
        )
        initial_coords = system.integrator_init(
            tuning_state.kinetic_state, state.q, p
        )
        initial_energy = state.potential_energy + system.kinetic(
            tuning_state.kinetic_state, state.q, p
        )

        final_coords, success = integrate(
            system,
            actual_num_steps,
            tuning_state.step_size,
            tuning_state.kinetic_state,
            initial_coords,
        )

        if isinstance(final_coords, LeapfrogState):
            potential_energy = system.potential(final_coords.q)
            potential_energy_grad = final_coords.dUdq
        else:
            potential_energy, potential_energy_grad = potential_and_grad(
                final_coords.q
            )
        kinetic_energy = system.kinetic(
            tuning_state.kinetic_state, final_coords.q, final_coords.p
        )
        energy = potential_energy + kinetic_energy

        delta = energy - initial_energy
        delta = jnp.where(jnp.isnan(delta), jnp.inf, delta)
        diverging = jnp.logical_not(success) | (delta > max_delta_energy)
        accept_prob = jnp.where(
            diverging, 0.0, jnp.clip(jnp.exp(-delta), a_max=1.0)
        )
        accept = random.bernoulli(rng_accept, accept_prob)

        # Construct the state at the endpoint that will be accepted or rejected
        # based on `accept` above
        final_state = HMCState(
            q=final_coords.q,
            potential_energy=potential_energy,
            potential_energy_grad=potential_energy_grad,
        )

        return SamplerCarry(
            tuning=tuning_state,
            state=jax.lax.cond(
                accept,
                lambda _: final_state,
                lambda _: state,
                operand=None,
            ),
            stats=ProposalStats(
                accept_prob=accept_prob,
                accept=accept,
                diverging=diverging,
            ),
        )

    def fast_update_fn(carry: SamplerCarry) -> SamplerCarry:
        tuning_state, state, stats = carry
        step_size_state = step_size_update_fn(
            tuning_state.step_size_state,
            target_accept_prob - stats.accept_prob,
        )
        return SamplerCarry(
            tuning=SamplerTuningState(
                step_size=jnp.exp(step_size_state.x),
                step_size_state=step_size_state,
                kinetic_state=system.kinetic_tune_update(
                    tuning_state.kinetic_state, state.q
                ),
            ),
            state=state,
            stats=stats,
        )

    def slow_update_fn(carry: SamplerCarry) -> SamplerCarry:
        tuning_state, state, stats = carry
        kinetic_state = system.kinetic_tune_update(
            tuning_state.kinetic_state, state.q
        )
        return SamplerCarry(
            tuning=SamplerTuningState(
                step_size=initial_step_size,
                step_size_state=step_size_init_fn(initial_step_size),
                kinetic_state=system.kinetic_tune_finish(kinetic_state),
            ),
            state=state,
            stats=stats,
        )

    def reset_fn(carry: SamplerCarry) -> SamplerCarry:
        _, state, stats = carry
        system.kinetic_tune_init(ravel_pytree(state.q)[0].size)
        return SamplerCarry(
            tuning=SamplerTuningState(
                step_size=initial_step_size,
                step_size_state=step_size_init_fn(
                    jnp.log(10 * initial_step_size)
                ),
                kinetic_state=system.kinetic_tune_init(
                    ravel_pytree(state.q)[0].size
                ),
            ),
            state=state,
            stats=stats,
        )

    def tune_finish_fn(carry: SamplerCarry) -> SamplerCarry:
        tuning_state, state, stats = carry
        step_size_state = step_size_update_fn(
            tuning_state.step_size_state,
            target_accept_prob - stats.accept_prob,
        )
        kinetic_state = system.kinetic_tune_update(
            tuning_state.kinetic_state, state.q
        )
        return SamplerCarry(
            tuning=SamplerTuningState(
                step_size=jnp.exp(step_size_state.x_avg),
                step_size_state=step_size_init_fn(initial_step_size),
                kinetic_state=system.kinetic_tune_finish(kinetic_state),
            ),
            state=state,
            stats=stats,
        )

    return MCMCKernel(
        init=init_fn,
        step=step_fn,
        reset=reset_fn,
        fast_update=fast_update_fn,
        slow_update=slow_update_fn,
        tune_finish=tune_finish_fn,
    )
