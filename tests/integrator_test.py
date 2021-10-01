from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from rmhmc.base_types import Array, Momentum, Position, Scalar
from rmhmc.hamiltonian import (
    System,
    compute_total_energy,
    integrate,
    integrate_trajectory,
)
from rmhmc.integrator import IntegratorState

from .problems import PROBLEMS


def run(
    system: System, num_steps: int, step_size: float, q: Position, p: Momentum
) -> Tuple[Scalar, Array, IntegratorState, Array]:
    kinetic_state = system.kinetic_tune_init(ravel_pytree(q)[0].size)
    state = system.integrator_init(kinetic_state, q, p)

    calc_energy = partial(compute_total_energy, system, kinetic_state)
    initial_energy = calc_energy(q, p)

    trace, success = integrate_trajectory(
        system, num_steps, step_size, kinetic_state, state
    )

    energy = jax.vmap(calc_energy)(trace.q, trace.p)
    return initial_energy, energy, trace, success


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_energy_conservation(problem_name: str) -> None:
    problem = PROBLEMS[problem_name]
    system = problem.builder()  # type: ignore
    initial_energy, energy, _, _ = jax.jit(
        partial(run, system, problem.num_steps, problem.step_size)
    )(problem.q, problem.p)
    np.testing.assert_allclose(
        energy, initial_energy, atol=problem.energy_prec
    )


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_reversibility(problem_name: str) -> None:
    problem = PROBLEMS[problem_name]
    system = problem.builder()  # type: ignore
    func = jax.jit(partial(run, system, problem.num_steps, problem.step_size))
    _, _, trace, _ = func(problem.q, problem.p)

    q = tree_map(lambda x_: x_[-1], trace.q)
    p = tree_map(lambda x_: -x_[-1], trace.p)
    _, _, rev_trace, _ = func(q, p)

    tree_map(
        lambda a, b: np.testing.assert_allclose(
            a[:-1][::-1], b[:-1], atol=problem.pos_prec
        ),
        trace.q,
        rev_trace.q,
    )


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_volume_conservation(problem_name: str) -> None:
    problem = PROBLEMS[problem_name]
    system = problem.builder()  # type: ignore

    q_flat, unravel = ravel_pytree(problem.q)
    p_flat, _ = ravel_pytree(problem.p)

    N = q_flat.size
    kinetic_state = system.kinetic_tune_init(N)
    phi = jax.jit(
        partial(
            integrate,
            system,
            problem.num_steps,
            problem.step_size,
            kinetic_state,
        )
    )

    qs = []
    ps = []

    eps = 1e-6
    for n in range(N):
        delta = 0.5 * eps * jnp.eye(N, 1, -n)[:, 0]

        q = unravel(q_flat + delta)
        plus, _ = phi(system.integrator_init(kinetic_state, q, problem.p))
        q = unravel(q_flat - delta)
        minus, _ = phi(system.integrator_init(kinetic_state, q, problem.p))
        qs.append((ravel_pytree(plus.q)[0] - ravel_pytree(minus.q)[0]) / eps)
        ps.append((ravel_pytree(plus.p)[0] - ravel_pytree(minus.p)[0]) / eps)

        p = unravel(p_flat + delta)
        plus, _ = phi(system.integrator_init(kinetic_state, problem.q, p))
        p = unravel(p_flat - delta)
        minus, _ = phi(system.integrator_init(kinetic_state, problem.q, p))
        qs.append((ravel_pytree(plus.q)[0] - ravel_pytree(minus.q)[0]) / eps)
        ps.append((ravel_pytree(plus.p)[0] - ravel_pytree(minus.p)[0]) / eps)

    F = jnp.concatenate(
        (jnp.stack(qs, axis=0), jnp.stack(ps, axis=0)), axis=-1
    )
    _, ld = jnp.linalg.slogdet(F)
    np.testing.assert_allclose(ld, 0.0, atol=problem.vol_prec)


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_integrate(problem_name: str) -> None:
    problem = PROBLEMS[problem_name]
    system = problem.builder()  # type: ignore

    kinetic_state = system.kinetic_tune_init(ravel_pytree(problem.q)[0].size)
    state = system.integrator_init(kinetic_state, problem.q, problem.p)
    final_state, success = jax.jit(
        partial(
            integrate,
            system,
            problem.num_steps,
            problem.step_size,
            kinetic_state,
        )
    )(state)
    assert success

    trajectory, success = jax.jit(
        partial(
            integrate_trajectory,
            system,
            problem.num_steps,
            problem.step_size,
            kinetic_state,
        )
    )(state)
    assert np.all(success)

    for v, t in zip(final_state, trajectory):
        tree_map(lambda a, b: np.testing.assert_allclose(a, b[-1]), v, t)
