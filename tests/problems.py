import dataclasses
from functools import partial
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from rmhmc.base_types import Array, Momentum, Position, Scalar
from rmhmc.hamiltonian import System, euclidean, riemannian


def sho(use_euclidean: bool) -> System:
    def log_posterior(q: Position) -> Scalar:
        return -0.5 * jnp.sum(q["x"] ** 2)

    def metric(q: Position) -> Array:
        return jnp.diag(jnp.ones_like(ravel_pytree(q)[0]))

    if use_euclidean:
        return euclidean(log_posterior)
    return riemannian(log_posterior, metric)


def planet(use_euclidean: bool) -> System:
    def log_posterior(q: Position) -> Scalar:
        return 1.0 / jnp.sqrt(jnp.sum(q ** 2))

    def metric(q: Position) -> Array:
        return jnp.diag(jnp.ones_like(q))

    if use_euclidean:
        return euclidean(log_posterior)
    return riemannian(log_posterior, metric)


def banana(fixed: bool, use_euclidean: bool) -> System:
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

    def log_posterior(q: Position) -> Scalar:
        p = q[0] + jnp.square(q[1])
        ll = jnp.sum(jnp.square(y - p)) / sigma_y ** 2
        lp = jnp.sum(jnp.square(theta)) / sigma_theta ** 2
        return -0.5 * (ll + lp)

    if fixed:

        def metric(q: Position) -> Array:
            return 10 * jnp.diag(jnp.ones_like(q))

    else:

        def metric(q: Position) -> Array:
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


@dataclasses.dataclass(frozen=True)
class Problem:
    builder: Callable[[], System]
    q: Position
    p: Momentum
    num_steps: int
    step_size: float
    energy_prec: float = 1e-4
    pos_prec: float = 5e-5
    vol_prec: float = 1e-5


PROBLEMS = dict(
    sho_riemannian=Problem(
        partial(sho, False),
        {"x": jnp.array([0.1])},
        {"x": jnp.array([2.0])},
        2000,
        0.01,
    ),
    sho_euclidean=Problem(
        partial(sho, True),
        {"x": jnp.array([0.1])},
        {"x": jnp.array([2.0])},
        2000,
        0.01,
    ),
    planet_riemannian=Problem(
        partial(planet, False),
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
        2000,
        0.01,
        pos_prec=5e-4,
        vol_prec=1e-4,
    ),
    planet_euclidean=Problem(
        partial(planet, True),
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
        2000,
        0.01,
    ),
    banana_riemannian=Problem(
        partial(banana, False, False),
        jnp.array([0.1, 0.3]),
        jnp.array([2.0, 0.5]),
        2000,
        0.001,
    ),
    banana_fixed=Problem(
        partial(banana, True, False),
        jnp.array([0.1, 0.3]),
        jnp.array([2.0, 0.5]),
        2000,
        0.001,
    ),
    banana_euclidean=Problem(
        partial(banana, True, True),
        jnp.array([0.1, 0.3]),
        jnp.array([2.0, 0.5]),
        2000,
        0.001,
        energy_prec=0.002,
    ),
)
