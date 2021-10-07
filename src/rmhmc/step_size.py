__all__ = ["dual_averaging"]

from typing import Callable, NamedTuple, Tuple

import jax.numpy as jnp

from .base_types import Scalar


class DualAveragingState(NamedTuple):
    count: Scalar
    x: Scalar
    x_avg: Scalar
    g_avg: Scalar
    mu: Scalar


def dual_averaging(
    t0: float = 10, kappa: float = 0.75, gamma: float = 0.05
) -> Tuple[
    Callable[[Scalar], DualAveragingState],
    Callable[[Scalar, DualAveragingState], DualAveragingState],
]:
    def init_fn(init: Scalar = 0.0) -> DualAveragingState:
        return DualAveragingState(
            count=jnp.array(0, dtype=jnp.result_type(int)),
            x=jnp.zeros(()),
            x_avg=jnp.zeros(()),
            g_avg=jnp.zeros(()),
            mu=init,
        )

    def update_fn(state: DualAveragingState, g: Scalar) -> DualAveragingState:
        count, _, x_avg, g_avg, mu = state
        count += 1
        g_avg = (1 - 1 / (count + t0)) * g_avg + g / (count + t0)
        x = mu - (count ** 0.5) / gamma * g_avg
        weight_t = count ** (-kappa)
        x_avg = (1 - weight_t) * x_avg + weight_t * x
        return DualAveragingState(
            count=count, x=x, x_avg=x_avg, g_avg=g_avg, mu=mu
        )

    return init_fn, update_fn
