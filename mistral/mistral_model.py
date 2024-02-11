from jax import Array
from transformers.models.mistral.modeling_mistral import MistralModel

from .decoder import DecoderParams, convert_decoder_params, forward_decoder, shard_decoder_params
from .embedding import EmbeddingParams, convert_embedding_params, forward_embedding, shard_embedding_params
from .rms_norm import RMSNormParams, convert_rms_norm_params, forward_rms_norm, shard_rms_norm_params

MistralModelParams = tuple[EmbeddingParams, DecoderParams, RMSNormParams]

def convert_mistral_model_params(model: MistralModel) -> MistralModelParams:
    embedding = convert_embedding_params(model.embed_tokens)
    decoder_layers = convert_decoder_params(model.layers)
    norm = convert_rms_norm_params(model.norm)
    return embedding, decoder_layers, norm

def convert_back_mistral_model_params():
    pass

def shard_mistral_model_params(params: MistralModelParams):
    embedding, decoder_layers, norm = params
    embedding = shard_embedding_params(embedding)
    decoder_layers = shard_decoder_params(decoder_layers)
    norm = shard_rms_norm_params(norm)
    return embedding, decoder_layers, norm

def forward_mistral_model(params: MistralModelParams, input_ids: Array, qk_mask: Array) -> Array:
    embedding, decoder_layers, norm = params
    seq = forward_embedding(embedding, input_ids)
    seq = forward_decoder(decoder_layers, seq, qk_mask)
    seq = forward_rms_norm(norm, seq)
    return seq

def test_forward_mistral_model():
    pass
