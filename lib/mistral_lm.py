from jax import Array
from transformers import MistralForCausalLM

from .array_conversion import pt2jax
from .mistral_model import MistralModelParams, convert_mistral_model_params, forward_mistral_model

MistralLMParams = tuple[MistralModelParams, Array]

def convert_mistral_lm_params(model: MistralForCausalLM) -> MistralLMParams:
    model_params = convert_mistral_model_params(model.model)
    lm_head = pt2jax(model.lm_head.weight.T)
    return model_params, lm_head

def convert_back_mistral_lm_params():
    pass

def forward_mistral_lm(params: MistralLMParams, input_ids: Array, qk_mask: Array) -> Array:
    model_params, lm_head = params

    outputs = forward_mistral_model(model_params, input_ids, qk_mask)
    logits = outputs @ lm_head
    return logits

def test_forward_mistral_lm():
    pass
