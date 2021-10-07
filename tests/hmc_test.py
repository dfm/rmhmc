import jax.numpy as jnp
import numpy as np
from jax import random

from rmhmc.hmc import hmc

from .problems import banana


def test_divergence() -> None:
    system = hmc(banana(False, False), initial_step_size=1000.0)

    state = system.init(jnp.array([0.3, 0.5]))
    state_ = system.step(state, random.PRNGKey(5))

    assert state_[2].diverging
    assert not state_[2].accept
    np.testing.assert_allclose(state_[2].accept_prob, 0.0)
