from jax import Array
import jax.numpy as jnp
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

def test_forward_mistral_lm(model: MistralForCausalLM) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
    tokenizer.pad_token = tokenizer.eos_token
    sentences = ['I have a cat.', 'There is a cat in my home.']
    inputs = tokenizer(sentences, padding=True, return_tensors='pt')
    input_ids = inputs.input_ids
    attn_mask = inputs.attention_mask

    outputs_pt = model(input_ids, attn_mask)[0]
    outputs_pt_to_jax = pt2jax(outputs_pt)

    params = convert_mistral_lm_params(model)
    input_ids_jax = pt2jax(input_ids)
    attn_mask_jax = pt2jax(attn_mask).astype(jnp.bool_)
    qk_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask_jax, attn_mask_jax))[:, None, None]
    outputs_jax = forward_mistral_lm(params, input_ids_jax, qk_mask)

    outputs_pt_to_jax = jnp.where(attn_mask_jax[:, :, None], outputs_pt_to_jax, 0.)
    outputs_jax = jnp.where(attn_mask_jax[:, :, None], outputs_jax, 0.)
    assert jnp.allclose(outputs_pt_to_jax, outputs_jax, atol=1e-5)
