from jax import Array
import jax.numpy as jnp

# shape (2 [fixed number] for k_cache and v_cache, 
# 32 [fixed number] for 32 blocks,
# 2 (dynamic) for batch_size,
# 8 [fixed number] for n_heads_kv，
# 9 (dynamic) for previous sentence length,
# 128 [fixed number] for dimension_n)
KVCache = Array | None
