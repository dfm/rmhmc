__all__ = ["run", "sample"]

import warnings
from functools import partial
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import lax, local_device_count, pmap, random, vmap
from jax._src.api import vmap
from jax.tree_util import tree_flatten

from rmhmc.base_types import Position, identity
from rmhmc.hmc import MCMCKernel, SamplerCarry


def sample(
    kernel: MCMCKernel,
    rng_key: random.KeyArray,
    initial_coords: Position,
    *,
    num_steps: int = 1000,
    num_tune: int = 1000,
    num_chains: int = 2,
    tune_kinetic_energy: bool = True,
    initial_buffer_size: int = 75,
    first_window_size: int = 25,
    final_buffer_size: int = 75,
    parallel: bool = True,
) -> SamplerCarry:
    def sample_one_chain(
        rng_key: random.KeyArray, q: Position
    ) -> SamplerCarry:
        tune_key, sample_key = random.split(rng_key)
        state = kernel.init(q)
        state, _ = run(
            kernel,
            num_tune,
            tune_key,
            state,
            tune=True,
            tune_kinetic_energy=tune_kinetic_energy,
            initial_buffer_size=initial_buffer_size,
            first_window_size=first_window_size,
            final_buffer_size=final_buffer_size,
        )
        return run(kernel, num_steps, sample_key, state, tune=False)[1]

    keys = random.split(rng_key, num_chains)
    if num_chains > 1:
        proto_init_val = tree_flatten(initial_coords)[0][0]
        if jnp.shape(proto_init_val)[0] != num_chains:
            raise ValueError(
                "The leading dimension of the initial parameters must match num_chains"
            )

    if parallel and num_chains >= local_device_count():
        warnings.warn(
            f"Only {local_device_count()} devices are visible to JAX; chains will be "
            "sampled sequentially"
        )
        parallel = False
    if parallel:
        execute = pmap(sample_one_chain)
    else:
        execute = vmap(sample_one_chain)

    return execute(keys, initial_coords)


def run(
    kernel: MCMCKernel,
    num_steps: int,
    rng_key: random.KeyArray,
    state: SamplerCarry,
    *,
    tune: bool = False,
    tune_kinetic_energy: bool = True,
    initial_buffer_size: int = 75,
    first_window_size: int = 25,
    final_buffer_size: int = 75,
) -> Tuple[SamplerCarry, SamplerCarry]:

    if tune:
        if tune_kinetic_energy:
            schedule = jnp.asarray(
                build_schedule(
                    num_steps,
                    initial_buffer_size=initial_buffer_size,
                    final_buffer_size=final_buffer_size,
                    first_window_size=first_window_size,
                )
            )
        else:
            schedule = jnp.zeros((num_steps, 2), dtype=bool)
            if initial_buffer_size <= 1:
                raise ValueError("'initial_buffer_size' must be >1")
            schedule[initial_buffer_size - 1, 1] = True

    else:
        schedule = jnp.zeros((num_steps, 2), dtype=bool)

    def step(
        state: SamplerCarry, args: Tuple[Tuple[bool, bool], random.KeyArray]
    ) -> Tuple[SamplerCarry, SamplerCarry]:
        (slow_update, reset), rng_key = args

        state_ = kernel.step(state, rng_key)

        # If we're tuning, update all the tuning parameters
        if tune:
            state_ = kernel.fast_update(state_)

            # Update the slow parameters if requested
            state_ = lax.cond(
                slow_update, kernel.slow_update, identity, state_
            )

            # Reset the tuning parameters if requested
            state_ = lax.cond(reset, kernel.reset, identity, state_)

        return state_, state_

    # state = system.init(q)
    state, trace = lax.scan(
        step, state, (schedule, random.split(rng_key, num_steps))
    )

    return kernel.tune_finish(state), trace


def build_schedule(
    num_steps: int,
    *,
    initial_buffer_size: int = 75,
    first_window_size: int = 25,
    final_buffer_size: int = 75,
) -> List[Tuple[bool, bool]]:
    num_steps = int(num_steps)
    initial_buffer_size = int(initial_buffer_size)
    first_window_size = int(first_window_size)
    final_buffer_size = int(final_buffer_size)

    if num_steps < 0:
        raise ValueError("'num_steps' must be >=0")
    if initial_buffer_size < 0:
        raise ValueError("'initial_buffer_size' must be >=0")
    if first_window_size < 1:
        raise ValueError("'first_window_size' must be >=1")
    if final_buffer_size < 0:
        raise ValueError("'final_buffer_size' must be >=0")

    # Special cases when num_steps is too small even for the hack below
    if num_steps == 0:
        warnings.warn("with zero tuning samples, the schedule is empty")
        return []

    if initial_buffer_size + first_window_size + final_buffer_size > num_steps:
        warnings.warn(
            "there are not enough tuning steps to accomodate the tuning "
            "schedule; assigning automatically as 20%/70%/10%"
        )
        initial_buffer_size = np.ceil(0.2 * num_steps).astype(int)
        final_buffer_size = np.ceil(0.1 * num_steps).astype(int)
        first_window_size = num_steps - initial_buffer_size - final_buffer_size

        # If this didn't cut it, 'num_steps' is too small (this should only happen
        # when num_steps == 1) just return one step of tuning
        if first_window_size <= 0:
            initial_buffer_size = 0
            final_buffer_size = 0
            first_window_size = num_steps

    t = initial_buffer_size
    delta = first_window_size
    if initial_buffer_size > 1:
        update_steps = [(False, False)] * (initial_buffer_size - 1) + [
            (False, True)
        ]
    else:
        update_steps = []
    while t < num_steps - final_buffer_size:
        if t + 2 * delta > num_steps - final_buffer_size:
            d = num_steps - final_buffer_size - t
            update_steps += [(False, False)] * (d - 1) + [(True, False)]
            break
        else:
            update_steps += [(False, False)] * (delta - 1) + [(True, False)]
        t += delta
        delta = 2 * delta

    if np.any(update_steps) <= 0:
        raise ValueError("invalid tuning schedule")

    return update_steps + [(False, False)] * final_buffer_size
