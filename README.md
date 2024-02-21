# Mistral JAX

## Installation

Simple installation from PyPI.

```sh
pip install mistral-jax
```

## Usage

```python
import jax
import jax.numpy as jnp
from mistral import convert_mistral_lm_params, forward_mistral_lm, shard_mistral_lm_params
from transformers import AutoTokenizer, MistralForCausalLM

model = MistralForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer.pad_token = tokenizer.eos_token

sentences = ['I have a cat.', 'There is a cat in my home.']
inputs = tokenizer(sentences, padding=True, return_tensors='jax')
input_ids = inputs.input_ids
attn_mask = inputs.attention_mask.astype(jnp.bool_)
qk_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None, None]

# load on CPU first to avoid OOM
cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    params = convert_mistral_lm_params(model)
params = shard_mistral_lm_params(params)

outputs = forward_mistral_lm(params, input_ids, qk_mask)
print(outputs)
```

## Roadmap

- [x] Model architecture
- [x] Publish a Python library
- [x] 1D Model parallelism
- [ ] Generation
    - [x] KV cache(batch_size = 1)
    - [ ] Left padding
    - [ ] Sampling
- [ ] Training

## Install

This project requires Python 3.11, JAX 0.4.20.

Create venv:

```sh
python3.11 -m venv venv
```

Install dependencies:

CPU:

```sh
pip install -U pip
pip install -U wheel
pip install "jax[cpu]"
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
```

CUDA 11:

```sh
pip install -U pip
pip install -U wheel
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
```

TPU VM:

```sh
pip install -U pip
pip install -U wheel
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
```

## Model architecture

```
MistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
            (self_attn): MistralAttention(
                    (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                    (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
                    (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
                    (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
                    (rotary_emb): MistralRotaryEmbedding()
            )
            (mlp): MistralMLP(
                    (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
                    (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
                    (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
                    (act_fn): SiLUActivation()
            )
            (input_layernorm): MistralRMSNorm()
            (post_attention_layernorm): MistralRMSNorm()
      )
    )
    (norm): MistralRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```
