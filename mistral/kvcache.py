from jax import Array
import jax.numpy as jnp

KVCache = tuple[list[list[Array] | None], list[list[Array] | None]] | None

def left_shift_kv_cache(kv_cache: KVCache) -> KVCache:
    # These 2 lists include cache for k and cache for v for 32 blocks.
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        # in the cache list, every cache shape is (batch_size, n_head, sentence padding length, dimension)
        k_cache = [jnp.roll(k_cache_blk, -1, axis=2) for k_cache_blk in k_cache]
        v_cache = [jnp.roll(v_cache_blk, -1, axis=2) for v_cache_blk in v_cache]
    return k_cache, v_cache
