from jax import Array
import jax.numpy as jnp
import torch
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from .attention import AttentionParams, convert_attention_params, forward_attention
from .decoder_block import DecoderBlockParams
from .rms_norm import RMSNormParams, convert_rms_norm_params, forward_rms_norm
from .mlp_layer import MLPLayerParams, convert_mlp_layer_params, forward_mlp_layer

DecoderBlockParams = tuple[RMSNormParams, AttentionParams, MLPLayerParams, RMSNormParams]

def convert_decoder_block_params(decoder_block: MistralDecoderLayer) -> DecoderBlockParams:
    input_layernorm = convert_rms_norm_params(decoder_block.input_layernorm)
    self_attn = convert_attention_params(decoder_block.self_attn)
    mlp = convert_mlp_layer_params(decoder_block.mlp)
    post_attention_layernorm = convert_rms_norm_params(decoder_block.post_attention_layernorm)
    return input_layernorm, self_attn, mlp, post_attention_layernorm

def convert_back_decoder_block_params():
    pass

def forward_decoder_block(params: DecoderBlockParams, seq: Array, qk_mask: Array) -> Array:
    input_layernorm, self_attn, mlp, post_attention_layernorm = params

    # residual connection
    seq_ = seq
    seq = forward_rms_norm(input_layernorm, seq)
    seq = forward_attention(self_attn, seq, qk_mask)
    seq += seq_

    seq_ = seq
    seq = forward_rms_norm(post_attention_layernorm, seq)
    seq = forward_mlp_layer(mlp, seq)
    seq += seq_

    return seq
