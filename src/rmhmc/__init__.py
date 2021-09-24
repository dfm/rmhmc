# -*- coding: utf-8 -*-

__all__ = []

from .rmhmc_version import __version__  # noqa

from . import hamiltonian, hmc, integrator

__uri__ = "https://github.com/dfm/rmhmc"
__author__ = "Dan Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__description__ = "Riemannian HMC in JAX"
__license__ = "MIT"
