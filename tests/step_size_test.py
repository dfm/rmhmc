from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from rmhmc.base_types import Array
from rmhmc.step_size import dual_averaging


def test_dual_averaging() -> None:
    def optimize(f: Callable[[Array], Array]) -> Array:
        init, update = dual_averaging(gamma=0.5)
        state = init(0.0)
        grad_fn = jax.grad(f)
        for i in range(10):
            g = grad_fn(state.x)
            state = update(state, g)
            print(state.x, state.x_avg)
        x_avg = state.x_avg
        return x_avg

    x_opt = optimize(lambda x: (x + 1) ** 2)
    np.testing.assert_allclose(x_opt, -1.0, atol=1e-3)
