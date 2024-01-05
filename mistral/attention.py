import math

import einops as op
import jax
from jax import Array
import jax.numpy as jnp
import torch
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralAttention

from .array_conversion import pt2jax
from .rotary_embedding import make_rotary_values, forward_rotary_embedding

# TODO: eliminate this
d_model = 4096
n_rep_kv = 4
n_heads_kv = 8
d_k = d_v = 128

AttentionParams = tuple[Array, Array, Array, Array]

def convert_attention_params(self_attn: MistralAttention) -> AttentionParams:
    q_proj = self_attn.q_proj.weight.data
    k_proj = self_attn.k_proj.weight.data
    v_proj = self_attn.v_proj.weight.data
    o_proj = self_attn.o_proj.weight.data
    
    q_proj_jax = pt2jax(q_proj.T).reshape(d_model, n_heads_kv, n_rep_kv, d_k).transpose(0, 2, 1, 3)
    k_proj_jax = pt2jax(k_proj.T).reshape(d_model, n_heads_kv, d_k)
    v_proj_jax = pt2jax(v_proj.T).reshape(d_model, n_heads_kv, d_v)
    out_proj_jax = pt2jax(o_proj.T).reshape(n_heads_kv, n_rep_kv, d_v, d_model).transpose(1, 0, 2, 3)

    return q_proj_jax, k_proj_jax, v_proj_jax, out_proj_jax

def convert_back_attention_params():
    pass

def forward_attention(params: AttentionParams, seq: Array, qk_mask: Array) -> Array:
    q_proj_jax, k_proj_jax, v_proj_jax, out_proj_jax = params

    # for q, the seq is src_seq, 
    # for k and v, the seq is des_seq,
    # in self_atten the src_ and des_seq are the same
    batch_size, seq_len, _ = seq.shape
    rotary_values = make_rotary_values(batch_size, seq_len)
    # q.shape: (1 batch_size, 4 n_rep_kv, 8 n_head, 6 seq_len ?, 128 k_dimension)
    # k.shape: (1 batch_size, 8 n_head, 6 seq_len, 128 k_dimension)
    # v.shape: (1 batch_size, 8 n_head, 6 seq_len, 128 v_dimension)

    # einsum can use to apply matrix multiplication
    q = op.einsum(seq, q_proj_jax, 'b s m, m r h k -> b r h s k')
    k = op.einsum(seq, k_proj_jax, 'b d m, m h k -> b h d k')
    v = op.einsum(seq, v_proj_jax, 'b d m, m h v -> b h d v')

    # before self attention, add position information
    # q.shape: (1 batch_size, 4, 8, 6 seq_len, 128)
    q = forward_rotary_embedding(q, rotary_values=rotary_values)
    k = forward_rotary_embedding(k, rotary_values=rotary_values)

    # self-attention
    # (1 batch_size, 4 repetition, 8 head number, 6 seq_len, 6 seq_len)
    # Scaled Dot-Product Attention as 3.2.1 equation(1) in orginal Transformer paper
    qk = jnp.einsum('brhsk,bhdk->brhsd', q, k) / math.sqrt(d_k)

    qk = jax.nn.softmax(qk, where=qk_mask, initial=0.)

    qkv = jnp.einsum('brhsd,bhdv->brhsv', qk, v)
    # (1, 4, 8, 6, 128)
    out = jnp.einsum('brhsv,rhvm->bsm', qkv, out_proj_jax)
    # out.shape: (1, 6, 4096); qkv (1, 4, 8, 6, 128); out_proj_jax.shape (4, 8, 128, 4096)
    return out

def test_forward_attention(model: MistralForCausalLM) -> None:
    batch_size = 1
    seq_len = 6

    self_attn_pt = model.model.layers[0].self_attn
    seq_pt = torch.rand(batch_size, seq_len, d_model, device=model.device)
    attention_mask_pt = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=model.device))
    attention_mask_pt_ = torch.where(attention_mask_pt, 0., -torch.inf)

    out_pt = self_attn_pt(seq_pt, attention_mask=attention_mask_pt_)[0]

    params = convert_attention_params(self_attn_pt)

    seq_jax = pt2jax(seq_pt)
    attention_mask_jax = pt2jax(attention_mask_pt)
    out_jax = forward_attention(params, seq_jax, attention_mask_jax)

    assert jnp.allclose(out_jax, pt2jax(out_pt), atol=1e-5)
