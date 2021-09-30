# type: ignore

from functools import partial
from typing import Any, Callable, Dict, NamedTuple, OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rmhmc.base_types import Momentum, Position
from rmhmc.hamiltonian import System, euclidean, riemannian


def sho(use_euclidean):
    def log_posterior(q):
        return -0.5 * jnp.sum(q ** 2)

    def metric(q):
        return jnp.diag(jnp.ones_like(q))

    if use_euclidean:
        return euclidean(log_posterior)
    return riemannian(log_posterior, metric)


def banana_problem(fixed, use_euclidean):
    t = 0.5
    sigma_y = 2.0
    sigma_theta = 2.0
    num_obs = 100

    random = np.random.default_rng(1234)
    theta = np.array([t, np.sqrt(1.0 - t)])
    y = (
        theta[0]
        + np.square(theta[1])
        + sigma_y * random.normal(size=(num_obs,))
    )

    def log_posterior(q):
        p = q[0] + jnp.square(q[1])
        ll = jnp.sum(jnp.square(y - p)) / sigma_y ** 2
        lp = jnp.sum(jnp.square(theta)) / sigma_theta ** 2
        return -0.5 * (ll + lp)

    if fixed:

        def metric(q):
            return 10 * jnp.diag(jnp.ones_like(q))

    else:

        def metric(q):
            n = y.size
            s = 2.0 * n * q[1] / sigma_y ** 2
            return jnp.array(
                [
                    [n / sigma_y ** 2 + 1.0 / sigma_theta ** 2, s],
                    [
                        s,
                        4.0 * n * jnp.square(q[1]) / sigma_y ** 2
                        + 1.0 / sigma_theta ** 2,
                    ],
                ]
            )

    if fixed and use_euclidean:
        return euclidean(log_posterior)
    return riemannian(log_posterior, metric)


class Problem(NamedTuple):
    builder: Callable[[], System]
    q: Position
    p: Momentum
    num_steps: int
    step_size: float


PROBLEMS = dict(
    sho_riemannian=Problem(
        partial(sho, False),
        jnp.array([0.1]),
        jnp.array([2.0]),
        2000,
        0.01,
    ),
    sho_euclidean=Problem(
        partial(sho, True),
        jnp.array([0.1]),
        jnp.array([2.0]),
        2000,
        0.01,
    ),
    banana_riemannian=Problem(
        partial(banana_problem, False, False),
        jnp.array([0.1, 0.3]),
        jnp.array([2.0, 0.5]),
        2000,
        0.001,
    ),
    banana_fixed=Problem(
        partial(banana_problem, True, False),
        jnp.array([0.1, 0.3]),
        jnp.array([2.0, 0.5]),
        2000,
        0.001,
    ),
    banana_euclidean=Problem(
        partial(banana_problem, True, True),
        jnp.array([0.1, 0.3]),
        jnp.array([2.0, 0.5]),
        2000,
        0.001,
    ),
)


def integrate_system(system, N, step_size, q, p):
    kinetic_state = system.kinetic_tune_init(q.size)
    calc_energy = lambda q_, p_: system.potential(q_) + system.kinetic(
        kinetic_state, q_, p_
    )
    state = system.integrator_init(kinetic_state, q, p)
    initial_energy = calc_energy(q, p)

    def step(state, _):
        update = system.integrator_update(
            kinetic_state, state, step_size=step_size
        )
        return update[0], update

    trace, success = jax.lax.scan(step, state, jnp.arange(N))[1]
    energy = jax.vmap(calc_energy)(trace.q, trace.p)
    return initial_energy, energy, trace, success


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_energy_conservation(problem_name):
    problem = PROBLEMS[problem_name]
    system = problem.builder()
    initial_energy, energy, _, _ = jax.jit(
        partial(integrate_system, system, problem.num_steps, problem.step_size)
    )(problem.q, problem.p)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(energy)
    plt.savefig(f"test_{problem_name}.png")

    np.testing.assert_allclose(energy, initial_energy, atol=2e-3)


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_reversibility(problem_name):
    problem = PROBLEMS[problem_name]
    system = problem.builder()
    func = jax.jit(
        partial(integrate_system, system, problem.num_steps, problem.step_size)
    )
    _, _, trace, _ = func(problem.q, problem.p)
    _, _, rev_trace, _ = func(trace.q[-1], -trace.p[-1])
    np.testing.assert_allclose(trace.q[:-1][::-1], rev_trace.q[:-1], atol=5e-5)
