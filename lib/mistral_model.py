from jax import Array
from transformers.models.mistral.modeling_mistral import MistralModel

from .embedding import EmbeddingParams, convert_embedding_params, forward_embedding
from .decoder import DecoderParams, convert_decoder_params, forward_decoder
from .rms_norm import RMSNormParams, convert_rms_norm_params, forward_rms_norm

MistralModelParams = tuple[EmbeddingParams, DecoderParams, RMSNormParams]

def convert_mistral_model_params(model: MistralModel) -> MistralModelParams:
    embedding = convert_embedding_params(model.embed_tokens)
    decoder_layers = convert_decoder_params(model.decoder_layers)
    norm = convert_rms_norm_params(model.norm)
    return embedding, decoder_layers, norm

def forward_mistral_model(params: MistralModelParams, input_ids: Array, qk_mask: Array) -> Array:
    embedding, decoder_layers, norm = params

    seq = forward_embedding(embedding, input_ids)
    seq = forward_decoder(decoder_layers, seq, qk_mask)
    seq = forward_rms_norm(norm, seq)
    return seq

def test_forward_mistral_model():
    pass
