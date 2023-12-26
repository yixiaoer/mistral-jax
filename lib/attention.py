import math

import einops as op
from jax import Array
import jax.numpy as jnp
import jax.nn as nn
from transformers.models.mistral.modeling_mistral import MistralAttention

from .array_conversion import pt2jax
from .temp_rotary_embedding import make_rotary_values, forward_rotary_embedding

# TODO: eliminate this
d_model = 4096
n_rep_kv = 4
n_heads_kv = 8
d_k = d_v = 128

AttentionParams = tuple[Array, Array, Array, Array]

def convert_attention_params(self_attn_torch: MistralAttention) -> AttentionParams:
    q_proj_torch = self_attn_torch.q_proj.weight.data
    k_proj_torch = self_attn_torch.k_proj.weight.data
    v_proj_torch = self_attn_torch.v_proj.weight.data
    o_proj_torch = self_attn_torch.o_proj.weight.data
    
    q_proj_jax = pt2jax(q_proj_torch.T).reshape(d_model, n_heads_kv, n_rep_kv, d_k).transpose(0, 2, 1, 3)
    k_proj_jax = pt2jax(k_proj_torch.T).reshape(d_model, n_heads_kv, d_k)
    v_proj_jax = pt2jax(v_proj_torch.T).reshape(d_model, n_heads_kv, d_v)
    out_proj_jax = pt2jax(o_proj_torch.T).reshape(n_heads_kv, n_rep_kv, d_v, d_model).transpose(1, 0, 2, 3)

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

    # q__1 = op.rearrange(q, 'b r h s k -> b (h r) s k')
    # q__2 = pt2jax(__temp_q)
    # assert jnp.allclose(q__1, q__2, atol=1e-5), (q__1, q__2)
    # assert jnp.allclose(k, pt2jax(__temp_k), atol=1e-5)
    # assert jnp.allclose(v, pt2jax(__temp_v), atol=1e-5)

    # before self attention, add position information
    # q.shape: (1 batch_size, 4, 8, 6 seq_len, 128)
    q = forward_rotary_embedding(q, rotary_values=rotary_values)
    k = forward_rotary_embedding(k, rotary_values=rotary_values)

    # q__1 = op.rearrange(q, 'b r h s k -> b (h r) s k')
    # q__2 = pt2jax(__temp_q_)
    # assert jnp.allclose(q__1, q__2, atol=1e-5), (q__1, q__2)
    # assert jnp.allclose(k, pt2jax(__temp_k_), atol=1e-5)

    # self-attention
    # (1 batch_size, 4 repetition, 8 head number, 6 seq_len, 6 seq_len)
    # Scaled Dot-Product Attention as 3.2.1 equation(1) in orginal Transformer paper
    qk = jnp.einsum('brhsk,bhdk->brhsd', q, k) / math.sqrt(d_k)
    # qk__1 = op.rearrange(qk, 'b r h s d -> b (h r) s d')
    # qk__2 = pt2jax(___scale_a_before)
    # assert jnp.allclose(qk__1, qk__2, atol=1e-5), (qk__1, qk__2)

    qk = nn.softmax(qk, where=qk_mask, initial=0.)
    # qk_ = op.rearrange(qk, 'b r h s d -> b (r h) s d')
    # print('qk_: ______', qk_)
    # print('__temp_qk: ______', __temp_qk)

    qkv = jnp.einsum('brhsd,bhdv->brhsv', qk, v)
    # (1, 4, 8, 6, 128)
    out = jnp.einsum('brhsv,rhvm->bsm', qkv, out_proj_jax)
    # out.shape: (1, 6, 4096); qkv (1, 4, 8, 6, 128); out_proj_jax.shape (4, 8, 128, 4096)
    return out

def test_forward_attention():
    import torch
    from transformers import MistralForCausalLM

    model = MistralForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1').to('cuda:1')  # JAX uses cuda:0

    batch_size = 1
    seq_len = 6

    self_attn_torch = model.model.layers[0].self_attn
    seq_torch = torch.rand(batch_size, seq_len, d_model, device='cuda:1')
    attention_mask_torch = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device='cuda:1'))
    attention_mask_torch_ = torch.where(attention_mask_torch, 0., -torch.inf)

    out_torch = self_attn_torch(seq_torch, attention_mask=attention_mask_torch_)[0]

    params = convert_attention_params(self_attn_torch)

    seq_jax = pt2jax(seq_torch)
    attention_mask_jax = pt2jax(attention_mask_torch)
    out_jax = forward_attention(params, seq_jax, attention_mask_jax)

    assert jnp.allclose(out_jax, pt2jax(out_torch), atol=1e-5)


# past_key_value = debug_dict['past_key_value']

# __temp_q = debug_dict['__temp_q']
# __temp_k = debug_dict['__temp_k']
# __temp_v = debug_dict['__temp_v']
# __temp_q_ = debug_dict['__temp_q_']
# __temp_k_ = debug_dict['__temp_k_']
# __temp_q_111 = debug_dict['__temp_q_111']
# __temp_k_111 = debug_dict['__temp_k_111']
# __temp_q_222 = debug_dict['__temp_q_222']
# __temp_k_222 = debug_dict['__temp_k_222']




# ___scale_a_before = debug_dict['___scale_a_before']
# ___scale_a = debug_dict['___scale_a']
# __temp_qk = debug_dict['__temp_qk']
# __temp_qkv = debug_dict['__temp_qkv']
# __temp_out = debug_dict['__temp_out']

# __temp_q.shape
# __temp_k.shape
# __temp_v.shape

# __temp_q_.shape
# __temp_k_.shape
# __temp_q_111.shape
# __temp_k_111.shape
# __temp_k_222.shape

# ___scale_a_before.shape
# ___scale_a.shape
# __temp_qk.shape

# __temp_qkv.shape
# __temp_out.shape




# batch_size = 1
# seq_len = 6

# self_attn_torch = model.model.layers[0].self_attn

# model.model.layers[0].self_attn.q_proj
# model.model.layers[0].self_attn.q_proj.weight
# q_proj_torch = self_attn_torch.q_proj.weight.data
# k_proj_torch = self_attn_torch.k_proj.weight.data
# v_proj_torch = self_attn_torch.v_proj.weight.data
# o_proj_torch = self_attn_torch.o_proj.weight.data

# # q_proj_torch.shape

# q_proj_jax = pt2jax(q_proj_torch).reshape(d_model, n_rep_kv, n_heads, d_k)
# k_proj_jax = pt2jax(k_proj_torch).reshape(d_model, n_heads, d_k)
# v_proj_jax = pt2jax(v_proj_torch).reshape(d_model, n_heads, d_v)
# out_proj_jax = pt2jax(o_proj_torch).reshape(n_rep_kv, n_heads, d_v, d_model)


# seq_torch = torch.rand(batch_size, seq_len, d_model)
# attention_mask_torch = torch.tril(torch.ones(
#     batch_size, 1, seq_len, seq_len, dtype=torch.bool))
# attention_mask_torch_ = torch.where(attention_mask_torch, 0., -torch.inf)

# out_torch = self_attn_torch(seq_torch, attention_mask=attention_mask_torch_)[0]


# seq_jax = pt2jax(seq_torch)
# attention_mask_jax = pt2jax(attention_mask_torch)
# out_jax = forward_attention(seq_jax, attention_mask_jax)
# out_jax
