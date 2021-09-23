__all__ = ["euclidean", "riemannian"]

from collections import namedtuple

from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

from .integrator import implicit_midpoint, velocity_verlet

System = namedtuple(
    "System",
    [
        "potential_fn",
        "kinetic_fn",
        "momentum_fn",
        "kinetic_init_fn",
        "kinetic_update_fn",
        "kinetic_finish_fn",
        "integrator_info_init",
        "integrator_init_fn",
        "integrator_update_fn",
    ],
)


def euclidean(log_probability_fn, diagonal=True):
    def potential_fn(q):
        return -log_probability_fn(q)

    def kinetic_fn(state, _, p):
        p, _ = ravel_pytree(p)
        tril = state[0]
        if tril.ndim == 1:
            alpha = tril * p
        else:
            alpha = jnp.dot(tril.T, p)
        return 0.5 * jnp.sum(jnp.square(alpha))

    def momentum_fn(state, q, rng_key):
        q, unravel = ravel_pytree(q)
        tril = state[0]
        eps = random.normal(rng_key, q.shape)
        if tril.ndim == 1:
            return unravel(eps / tril)
        return unravel(jsp.linalg.solve_triangular(tril, eps, trans=1, lower=True))

    def init_fn(size):
        shape = (size,) if diagonal else (size, size)
        mu = jnp.zeros(shape[-1])
        m2 = jnp.zeros(shape)
        return jnp.eye(size), 0, mu, m2

    def update_fn(q, state):
        q = ravel_pytree(q)
        tril, n, mu, m2 = state
        n += 1
        d1 = q - mu
        mu += d1 / n
        d2 = q - mu
        if m2.ndim == 1:
            m2 += d1 * d2
        else:
            m2 += jnp.outer(d2, d1)
        return tril, n, mu, m2

    def finish_fn(state):
        n, mu, m2 = state
        cov = m2 / (n - 1)
        if m2.ndim == 2:
            tril = jsp.linalg.cholesky(cov, lower=True)
        else:
            tril = jnp.sqrt(cov)
        return tril, 0, jnp.zeros_like(mu), jnp.zeros_like(m2)

    return System(
        potential_fn,
        kinetic_fn,
        momentum_fn,
        init_fn,
        update_fn,
        finish_fn,
        *velocity_verlet(potential_fn, kinetic_fn),
    )


def riemannian(log_probability_fn, metric_fn):
    def potential_fn(q):
        return -log_probability_fn(q)

    def kinetic_fn(_, q, p):
        p, _ = ravel_pytree(p)
        metric = metric_fn(q)
        tril, _ = jsp.linalg.cho_factor(metric, lower=True)
        half_log_det = jnp.sum(jnp.log(jnp.diag(tril)))
        alpha = jsp.linalg.solve_triangular(tril, p, lower=True)
        return 0.5 * jnp.sum(jnp.square(alpha)) + p.size * half_log_det

    def momentum_fn(_, q, rng_key):
        metric = metric_fn(q)
        q, unravel = ravel_pytree(q)
        metric = metric_fn(q)
        tril, _ = jsp.linalg.cho_factor(metric, lower=True)
        eps = random.normal(rng_key, q.shape)
        return unravel(jnp.dot(tril, eps))

    # This metric doesn't have any tuning parameters
    def init_fn(_):
        return None

    def update_fn(*_):
        return None

    def finish_fn(_, regularize=False):
        return None

    return System(
        potential_fn,
        kinetic_fn,
        momentum_fn,
        init_fn,
        update_fn,
        finish_fn,
        *implicit_midpoint(potential_fn, kinetic_fn),
    )
