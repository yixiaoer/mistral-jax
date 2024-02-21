from jax import Array

KVCache = tuple[list[list[Array] | None], list[list[Array] | None]] | None
