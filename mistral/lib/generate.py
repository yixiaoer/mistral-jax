from typing import Callable

from jax import Array
from jax.nn import softmax
import jax.numpy as jnp
from transformers import AutoTokenizer, MistralForCausalLM

from ..model.kvcache import KVCache, left_shift_kv_cache
from ..model.mistral_lm import MistralLMParams, forward_mistral_lm
from ..model.rotary_embedding import get_rotary_values_at_position, make_rotary_values

def generate(params: MistralLMParams, sentences: list[str], tokenizer: AutoTokenizer, max_length: int, max_new_tokens: int, sample_fn: Callable) -> Array:
    inputs = tokenizer(sentences, padding='max_length', max_length=max_length, return_tensors='jax')
    input_ids = inputs.input_ids.astype(jnp.uint16)
    output_ids = input_ids
    batch_size = len(input_ids)
    attn_mask = inputs.attention_mask.astype(jnp.bool_)
    qk_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None, None]
    kv_cache = None
    
    padding_len = jnp.sum(attn_mask == True, axis=1)
    num_generate_tokens = min(*(max_length - padding_len), max_new_tokens)

    rotary_values_cur = make_rotary_values(padding_len, batch_size, max_length)
    rotary_values = make_rotary_values(padding_len, batch_size, max_length)

    for idx in range(num_generate_tokens):
        if idx != 0:
            position = jnp.array((idx - 1), jnp.int16)
            rotary_values_cur = get_rotary_values_at_position(rotary_values, position)
        logits, kv_cache = forward_mistral_lm(params, input_ids, qk_mask, rotary_values_cur, kv_cache)

        next_prob = softmax(logits[:, -1])
        next_token = sample_fn(next_prob)
        
        input_ids = next_token
        output_ids = jnp.concatenate((output_ids[:,1:], input_ids), axis=1)
        attn_mask = jnp.concatenate((attn_mask[:,1:], jnp.ones((batch_size, 1), dtype=bool)), axis=1)
        qk_mask = attn_mask[:, None, None, None, :]
        kv_cache = left_shift_kv_cache(kv_cache)
    return output_ids

def greedy_search(probs: Array) -> Array:
    return jnp.argmax(probs, axis=-1).reshape(-1,1)
