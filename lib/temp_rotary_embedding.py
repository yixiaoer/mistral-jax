# from typing import NamedTuple

# import einops as op
# from jax import Array
# import jax.numpy as jnp


# def _make_weights(seq_len: int, d_k: int) -> tuple[Array, Array]:
#     inv_freq = 1. / (10000 ** (jnp.arange(0, d_k, 2) / d_k))
#     sinusoid_inp = op.einsum(jnp.arange(seq_len), inv_freq, 'L, j -> L j')
#     sin_val = jnp.sin(sinusoid_inp)
#     cos_val = jnp.cos(sinusoid_inp)
#     sin_val = op.repeat(sin_val, 'L K -> L (i K)', i=2)
#     cos_val = op.repeat(cos_val, 'L K -> L (i K)', i=2)
#     return sin_val, cos_val


# def _rotate_half(x: Array) -> Array:
#     # split the last dimension: (..., n) -> (..., 2, n // 2)
#     x = op.rearrange(x, '... (i x) -> ... i x', i=2)
#     x = x[..., ::-1, :]  # reverse dimension -2
#     x = x.at[..., 0, :].multiply(-1)  # negate the first half of dimension -2
#     # merge the last two dimensions: (..., 2, n // 2) -> (..., n)
#     x = op.rearrange(x, '... i x -> ... (i x)')
#     return x


# class RotaryValues(NamedTuple):
#     sin_val: Array
#     cos_val: Array


# def forward_rotary_embedding(m: Array, *, rotary_values: RotaryValues) -> Array:
#     sin_val, cos_val = rotary_values
#     assert sin_val.dtype == jnp.float32
#     assert cos_val.dtype == jnp.float32
#     n = _rotate_half(m)
#     # for q
#     # m.shape/n.shape: (1 batch_size, 4 n_rep_kv, 32, 6 seq_len, 32)
#     # sin_val.shape/cos_val.shape: (1 batch_size, 6 seq_len, 32)
#     # for v
#     # m.shape/n.shape: (1 batch_size, 32, 6 seq_len, 32)
#     # sin_val.shape/cos_val.shape: (1 batch_size, 6 seq_len, 32)
#     a = op.einsum(m, cos_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
#     b = op.einsum(n, sin_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
#     return a + b


# def make_rotary_values(seq_len: int) -> RotaryValues:
#     sin_val, cos_val = _make_weights(seq_len, d_k)
#     # values are the same, add another dimension
#     # the same function as sin[position_ids].unsqueeze(unsqueeze_dim) in PyTorch
#     sin_val = jnp.repeat(sin_val[None], 1, axis=0)
#     cos_val = jnp.repeat(cos_val[None], 1, axis=0)
#     return RotaryValues(sin_val, cos_val)



from typing import NamedTuple

import einops as op
import jax
from jax import Array
import jax.numpy as jnp

# # TODO: eliminate this
d_k = 128

# TODO: Mostly taken from https://github.com/kingoflolz/mesh-transformer-jax/blob/master/mesh_transformer/layers.py
# and https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L92
def _make_weights(seq_len: int, d_k: int) -> tuple[Array, Array]:
    inv_freq = 1. / (10000 ** (jnp.arange(0, d_k, 2) / d_k))
    sinusoid_inp = op.einsum(jnp.arange(seq_len), inv_freq, 'L, j -> L j')
    sin_val = jnp.sin(sinusoid_inp)
    cos_val = jnp.cos(sinusoid_inp)
    sin_val = op.repeat(sin_val, 'L K -> L (i K)', i=2)
    cos_val = op.repeat(cos_val, 'L K -> L (i K)', i=2)
    return sin_val, cos_val

def _rotate_half(x: Array) -> Array:
    x = op.rearrange(x, '... (i x) -> ... i x', i=2)  # split the last dimension: (..., n) -> (..., 2, n // 2)
    x = x[..., ::-1, :]  # reverse dimension -2
    x = x.at[..., 0, :].multiply(-1)  # negate the first half of dimension -2
    x = op.rearrange(x, '... i x -> ... (i x)')  # merge the last two dimensions: (..., 2, n // 2) -> (..., n)
    return x

class RotaryValues(NamedTuple):
    sin_val: Array
    cos_val: Array

def forward_rotary_embedding(m: Array, *, rotary_values: RotaryValues) -> Array:
    sin_val, cos_val = rotary_values
    assert sin_val.dtype == jnp.float32
    assert cos_val.dtype == jnp.float32
    n = _rotate_half(m)
    a = op.einsum(m, cos_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
    b = op.einsum(n, sin_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
    return a + b

def make_rotary_values(batch_size: int, seq_len: int) -> RotaryValues:
    sin_val, cos_val = _make_weights(seq_len, d_k)

    sin_val = jnp.repeat(sin_val[None], batch_size, axis=0)
    cos_val = jnp.repeat(cos_val[None], batch_size, axis=0)

    return RotaryValues(sin_val, cos_val)

def get_rotary_values_at_position(rotary_values: RotaryValues, position: Array) -> RotaryValues:
    sin_val, cos_val = rotary_values
    sin_val = sin_val[:, position][:, None]
    cos_val = cos_val[:, position][:, None]
    rotary_values = RotaryValues(sin_val, cos_val)
    return rotary_values
