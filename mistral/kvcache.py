from jax import Array
import jax.numpy as jnp

KVCache = tuple[list[list[Array] | None], list[list[Array] | None]] | None

def left_shift_kv_cache(kv_cache: KVCache) -> KVCache:
    # KVCache tuple contains 2 lists: k_cache and v_cache for 32 blocks.
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        k_cache = [jnp.roll(k_cache_blk, -1, axis=2) for k_cache_blk in k_cache]
        v_cache = [jnp.roll(v_cache_blk, -1, axis=2) for v_cache_blk in v_cache]
    return k_cache, v_cache
