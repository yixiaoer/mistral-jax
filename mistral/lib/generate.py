from typing import Callable

from jax import Array, lax
from jax.nn import softmax
import jax.numpy as jnp
import jax.random as jrand
from transformers import AutoTokenizer, MistralForCausalLM

from ..model.kvcache import KVCache
from ..model.mistral_lm import MistralLMParams, forward_mistral_lm
from ..model.rotary_embedding import RotaryValues, get_rotary_values_at_position, make_rotary_values

def generate(params: MistralLMParams, tokenizer: AutoTokenizer, sentences: list[str], max_length: int, max_new_tokens: int, *, key: Array | None = None, top_k: int | None = None, top_p: float | None = None, temperature: float = 1.) -> Array:
    # `max_length` and `max_new_tokens` jointly influence the maximum number of tokens generated
    inputs = tokenizer(sentences, padding=True, return_tensors='jax')
    input_ids = inputs.input_ids.astype(jnp.uint16)
    output_ids = input_ids
    batch_size, batch_len = input_ids.shape
    attn_mask = inputs.attention_mask.astype(jnp.bool_)
    qk_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None, None]
    kv_cache = None

    generate_tokens_n = min(*(max_length - jnp.sum(attn_mask == True, axis=1)), max_new_tokens)
    rotary_values_cur = make_rotary_values(batch_size, batch_len)
    rotary_values = make_rotary_values(batch_size, batch_len + generate_tokens_n)
    for idx in range(generate_tokens_n):
        if idx != 0:
            position = jnp.array((batch_len + idx - 1), jnp.int16)
            rotary_values_cur = get_rotary_values_at_position(rotary_values, position)
        logits, kv_cache = forward_mistral_lm(params, input_ids, qk_mask, rotary_values_cur, kv_cache)
        logits = logits[:, -1]

        if (top_k is None or top_k == 1) and (temperature == 0. or temperature == 1.) and top_p ==  None:
            next_token = greedy_search(logits)
        else:
            # first top_K, then top_p
            if top_k:
                logits, tokens_ids = top_k_logits(logits, temperature=temperature, top_k=top_k)
            if top_p:
                temperature, tokens_ids = (1., tokens_ids) if top_k else (temperature, None)
                logits, tokens_ids = top_p_logits(logits, tokens_ids=tokens_ids, temperature=temperature, top_p=top_p)
            next_token = sampling(logits, tokens_ids, key)

        input_ids = next_token
        output_ids = jnp.concatenate((output_ids, next_token), axis=1)
        attn_mask = jnp.concatenate((attn_mask, jnp.ones((batch_size, 1), dtype=bool)), axis=1)
        qk_mask = attn_mask[:, None, None, None, :]
    return output_ids

def greedy_search(logits: Array) -> Array:
    return jnp.argmax(logits, axis=-1).reshape(-1, 1)

def top_k_logits(logits: Array, *, temperature: float = 1.0, top_k: int) -> tuple[Array, Array]:
    top_k_logits, top_k_ids = lax.top_k(logits, top_k)  # along the last axis
    return top_k_logits / temperature, top_k_ids

def top_p_logits(logits: Array, *, tokens_ids: Array | None = None, temperature: float = 1.0, top_p: float) -> tuple[Array, Array]:
    # cumulative probability of the generated tokens
    if tokens_ids is None:
        tokens_ids = jnp.argsort(-logits, axis=-1)
        logits = jnp.take_along_axis(logits, tokens_ids, axis=-1)

    probs_sorted_cumsum = jnp.cumsum(softmax(logits, axis=-1), axis=-1)
    cutoff_index = jnp.sum(probs_sorted_cumsum < top_p, axis=-1).reshape(-1, 1)
    logits = jnp.where(jnp.arange(logits.shape[-1])[None, :] <= cutoff_index, logits, -jnp.inf)
    return logits / temperature, tokens_ids

def sampling(sampling_logits: Array, tokens_ids: Array, key: Array | None = None) -> Array:
    sampling_index = jrand.categorical(key, softmax(sampling_logits)).reshape(-1, 1)
    selected_token = jnp.take_along_axis(tokens_ids, sampling_index, axis=-1)
    return selected_token
