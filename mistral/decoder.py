
from jax import Array
from torch.nn import ModuleList as TorchModuleList

from .decoder_block import DecoderBlockParams, convert_decoder_block_params, forward_decoder_block, shard_decoder_block_params
from .einshard import einshard

DecoderParams = list[DecoderBlockParams]

def convert_decoder_params(layers: TorchModuleList) -> DecoderParams:
    return [convert_decoder_block_params(layer) for layer in layers]

def convert_back_decoder_params():
    raise NotImplementedError

def shard_decoder_params(layers: DecoderParams) -> DecoderParams:
    return [shard_decoder_block_params(layer) for layer in layers]

def forward_decoder(params: DecoderParams, seq: Array, qk_mask: Array) -> Array:
    # TODO: jax.lax.scan
    for param in params:
        seq = forward_decoder_block(param, seq, qk_mask)
    return seq

def test_forward_decoder():
    pass
