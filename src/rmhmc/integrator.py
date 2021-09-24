__all__ = ["velocity_verlet", "implicit_midpoint"]

from collections import namedtuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

VerletState = namedtuple("VerletState", ["q", "p", "dUdq"])
ImplicitState = namedtuple("ImplicitState", ["q", "p", "dHdq", "dHdp"])
SolverInfo = namedtuple("SolverInfo", ["iterations", "success"])


def velocity_verlet(potential_fn, kinetic_fn):
    dU = jax.grad(potential_fn)

    def init_fn(_, q, p):
        return VerletState(q, p, dU(q)), None

    def update_fn(kinetic_state, state, *, epsilon):
        p = tree_map(
            lambda p, dUdq: p - 0.5 * epsilon * dUdq, state.p, state.dUdq
        )
        dTdp = jax.grad(kinetic_fn, argnums=2)(kinetic_state, None, p)
        q = tree_map(lambda q, dTdp: q + epsilon * dTdp, state.q, dTdp)
        dUdq = dU(q)
        p = tree_map(lambda p, dUdq: p - 0.5 * epsilon * dUdq, p, dUdq)
        return VerletState(q, p, dUdq), None

    return None, init_fn, update_fn


def implicit_midpoint(potential_fn, kinetic_fn, **solver_kwargs):
    hamiltonian = lambda state, q, p: potential_fn(q) + kinetic_fn(state, q, p)
    vector_field = jax.grad(hamiltonian, argnums=(1, 2))
    info_init = dict(iterations=0, success=True)

    def init_fn(state, q, p):
        dHdq, dHdp = vector_field(state, q, p)
        return ImplicitState(q, p, dHdq, dHdp), info_init

    def update_fn(kinetic_state, value, *, epsilon):
        state, info0 = value

        def step(args):
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
        (q, p), info = solve_fixed_point(step, (q, p), **solver_kwargs)

        # Compute the resulting vector field and update the state
        dHdq = (2.0 / epsilon) * (state.p - p)
        dHdp = (2.0 / epsilon) * (q - state.q)
        q = 2 * q - state.q
        p = 2 * p - state.p

        return ImplicitState(q, p, dHdq, dHdp), dict(
            iterations=info0["iterations"] + info.iterations,
            success=info0["success"] & info.success,
        )

    return info_init, init_fn, update_fn


def solve_fixed_point(
    func,
    x0,
    convergence_tol=1e-6,
    divergence_tol=1e10,
    max_iters=100,
    norm_fn=lambda x: jnp.max(jnp.abs(x)),
):
    def compute_norm(x, xp):
        return norm_fn(ravel_pytree(tree_map(jnp.subtract, x, xp))[0])

    def cond_fn(args):
        n, _, norm = args
        return (
            (n < max_iters)
            & jnp.isfinite(norm)
            & (norm < divergence_tol)
            & (norm > convergence_tol)
        )

    def body_fn(args):
        n, x, _ = args
        xn = func(x)
        norm = compute_norm(xn, x)
        return n + 1, xn, norm

    x = func(x0)
    n, x, norm = jax.lax.while_loop(
        cond_fn, body_fn, (0, x, compute_norm(x, x0))
    )
    success = jnp.isfinite(norm) & (norm <= convergence_tol)
    return x, SolverInfo(iterations=n, success=success)
