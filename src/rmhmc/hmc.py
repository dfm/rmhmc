# __all__ = [""]

# from collections import namedtuple

# import jax
# import jax.numpy as jnp
# from jax import random
# from jax.flatten_util import ravel_pytree

# from .hamiltonian import System

# HMCState = namedtuple(
#     "HMCState",
#     [
#         "q",
#         "potential_energy",
#         "accept_prob",
#         "accept",
#         "diverging",
#         "step_size",
#         "sample_stats",
#         "rng_key",
#         "kinetic_state",
#         "step_size_state",
#     ],
# )


# def hmc(system: System, rng_key, num_steps=50):
#     step_size_init_fn, step_size_update_fn = dual_averaging()
#     max_delta_energy = 1000.0

#     def init_fn(q):
#         return HMCState(
#             q=q,
#             potential_energy=system.potential_fn(q),
#             accept_prob=0.0,
#             accept=False,
#             diverging=False,
#             step_size=0.001,
#             sample_stats=system.integrator_info_init,
#             rng_key=rng_key,
#             kinetic_state=system.kinetic_init_fn(ravel_pytree(q)[0].size),
#             step_size_state=step_size_init_fn(),
#         )

#     def update_fn(state):
#         rng_mom, rng_accept, rng_next = random.split(state.rng_key, 3)

#         p = system.momentum_fn(state.kinetic_state, state.q, rng_mom)
#         initial_coords = system.integrator_init_fn(
#             state.kinetic_state, state.q, p
#         )
#         initial_energy = state.potential_energy + system.kinetic_fn(
#             state.kinetic_state, state.q, p
#         )

#         final_coords, info = jax.lax.fori_loop(
#             0,
#             num_steps,
#             lambda _, coords: system.integrator_update_fn(
#                 state.kinetic_state, coords, step_size=state.step_size
#             ),
#             initial_coords,
#         )

#         potential_energy = system.potential_fn(final_coords.q)
#         kinetic_energy = system.kinetic_fn(
#             state.kinetic_state, final_coords.q, final_coords.p
#         )
#         energy = potential_energy + kinetic_energy

#         delta = energy - initial_energy
#         delta = jnp.where(jnp.isnan(delta), jnp.inf, delta)
#         diverging = delta > max_delta_energy
#         accept_prob = jnp.clip(jnp.exp(-delta), a_max=1.0)
#         accept = random.bernoulli(rng_accept, accept_prob)

#         new_state = HMCState(
#             q=final_coords.q,
#             potential_energy=potential_energy,
#             accept_prob=accept_prob,
#             accept=accept,
#             diverging=diverging,
#             step_size=state.step_size,
#             sample_stats=info,
#             rng_key=rng_next,
#             kinetic_state=state.kinetic_state,
#             step_size_state=state.step_size_state,
#         )

#         return jax.lax.cond(
#             accept, lambda _: new_state, lambda _: state, operand=None
#         )

#     return init_fn, update_fn


# def dual_averaging(t0=10, kappa=0.75, gamma=0.05):
#     def init_fn(prox_center=0.0):
#         x_t = jnp.zeros(())
#         x_avg = jnp.zeros(())
#         g_avg = jnp.zeros(())
#         t = jnp.array(0, dtype=jnp.result_type(int))
#         return x_t, x_avg, g_avg, t, prox_center

#     def update_fn(g, state):
#         x_t, x_avg, g_avg, t, prox_center = state
#         t = t + 1
#         g_avg = (1 - 1 / (t + t0)) * g_avg + g / (t + t0)
#         x_t = prox_center - (t ** 0.5) / gamma * g_avg
#         weight_t = t ** (-kappa)
#         x_avg = (1 - weight_t) * x_avg + weight_t * x_t
#         return x_t, x_avg, g_avg, t, prox_center

#     return init_fn, update_fn
