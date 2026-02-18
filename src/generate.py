"""Backward-compatibility shim — all generators now live in src.generators."""
from .generators import *  # noqa: F401,F403
from .generators import GENERATOR_REGISTRY, available_generators, generate_series
