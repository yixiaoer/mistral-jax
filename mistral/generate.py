from typing import Callable

from jax import Array
import jax.numpy as jnp
from jax.nn import softmax
from transformers import AutoTokenizer, MistralForCausalLM

from .kvcache import KVCache
from .mistral_lm import MistralLMParams, forward_mistral_lm
from .rotary_embedding import get_rotary_values_at_position, make_rotary_values

def generate(params: MistralLMParams, sentences: list[str], tokenizer: AutoTokenizer, max_new_tokens: int, sample_fn: Callable) -> Array:
    # TODO: one sentence with bacth_size=1, for batch_size > 1, deal with `left padding` later
    inputs = tokenizer(sentences, padding=True, return_tensors='jax')
    input_ids = inputs.input_ids.astype(jnp.uint16)
    output_ids = input_ids
    batch_size, input_len = input_ids.shape

    attn_mask = inputs.attention_mask.astype(jnp.bool_)
    qk_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None, None]
    kv_cache_cur, kv_cache_pre = None, None

    rotary_values_cur = make_rotary_values(batch_size, input_len)
    rotary_values = make_rotary_values(batch_size, input_len + max_new_tokens)

    for idx in range(max_new_tokens):
        if idx != 0:
            position = jnp.array((input_len + idx), jnp.int16)
            rotary_values_cur = get_rotary_values_at_position(rotary_values, position)
        logits, kv_cache_cur, kv_cache_pre = forward_mistral_lm(params, input_ids, qk_mask, rotary_values_cur, kv_cache_cur, kv_cache_pre)

        logits = logits[0]
        next_prob = softmax(logits[-1])
        next_token = sample_fn(next_prob)

        input_ids = next_token.reshape(1,-1)
        output_ids = jnp.concatenate((output_ids, input_ids), axis=1)
        qk_mask = None
    return output_ids

def greedy_search(probs: Array) -> Array:
    # select the word with the highest probability
    return jnp.argmax(probs)
