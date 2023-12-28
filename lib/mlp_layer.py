import jax
from jax import Array
import jax.numpy as jnp
import torch
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralMLP

from .array_conversion import pt2jax

MLPLayerParams = tuple[Array, Array, Array]

def convert_mlp_layer_params(mlp_layer: MistralMLP) -> MLPLayerParams:
    gate_proj = pt2jax(mlp_layer.gate_proj.weight.data.T)
    up_proj = pt2jax(mlp_layer.up_proj.weight.data.T)
    down_proj = pt2jax(mlp_layer.down_proj.weight.data.T)
    return gate_proj, up_proj, down_proj

def convert_back_mlp_layer_params():
    pass

def forward_mlp_layer(params: MLPLayerParams, seq: Array) -> Array:
    gate_proj, up_proj, down_proj = params

    ff = jax.nn.silu(seq @ gate_proj) * (seq @ up_proj)
    seq = ff @ down_proj
    return seq

def test_forward_mlp_layer(model: MistralForCausalLM) -> None:
    pass
