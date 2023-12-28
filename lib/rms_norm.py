import jax
from jax import Array
import jax.numpy as jnp
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from .array_conversion import pt2jax

# TODO: eliminate this
d_model = 4096

rms_norm_eps = 1e-5

RMSNormParams = Array

def convert_rms_norm_params(rms_norm: MistralRMSNorm) -> RMSNormParams:
    return pt2jax(rms_norm.weight)

def convert_back_rms_norm_params():
    pass

# Taken from https://github.com/ayaka14732/llama-2-jax/blob/main/lib/llama/rms_norm.py
def forward_rms_norm(params: RMSNormParams, x: Array) -> Array:
    x_rms = jnp.sqrt((x * x).mean(axis=-1, keepdims=True) + rms_norm_eps)
    y = x / x_rms * params
    return y
