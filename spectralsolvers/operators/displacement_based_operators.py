import jax  # type: ignore

jax.config.update("jax_compilation_cache_dir", "/cluster/scratch/mpundir/jax-cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp  # type: ignore

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")
import numpy as np
import functools

import itertools

# Dirac delta function
δ = lambda i, j: float(i == j)