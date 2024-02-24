from typing import Callable

import jax
import torch
from transformers import AutoTokenizer, MistralForCausalLM

from mistral.mistral_lm import convert_mistral_lm_params, shard_mistral_lm_params
from mistral.generate import generate, greedy_search

def main():
    model = MistralForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
    tokenizer.pad_token = tokenizer.eos_token

    # load on CPU first to avoid OOM
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        params = convert_mistral_lm_params(model)
    params = shard_mistral_lm_params(params)

    sentences = ['How have you been?', 'The Lord of the Rings is a']
    max_new_tokens = 32
    max_length = 64

    output_ids = generate(params, sentences, tokenizer, max_length, max_new_tokens, greedy_search)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    inputs_pt = tokenizer(sentences, padding='max_length', max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        generated_pt = model.generate(input_ids=inputs_pt.input_ids, attention_mask=inputs_pt.attention_mask, do_sample=False, max_new_tokens=max_new_tokens)
    output_pt = tokenizer.batch_decode(generated_pt, skip_special_tokens=True)

    print(f'JAX output: {output}')
    print(f'PyTorch output: {output_pt}')
    print(f'JAX output == PyTorch output: {output == output_pt}')
    # JAX output: ['How have you been? I’ve been busy with work and life, but I’m still here. I’m still here.\n\nI’ve been thinking about this', 'The Lord of the Rings is a series of three epic fantasy novels written by J. R. R. Tolkien. The books tell of the quest of a group of heroes to destroy a']
    # PyTorch output: ['How have you been? I’ve been busy with work and life, but I’m still here. I’m still here.\n\nI’ve been thinking about this', 'The Lord of the Rings is a series of three epic fantasy novels written by J. R. R. Tolkien. The books tell of the quest of a group of heroes to destroy a']
    # JAX output == PyTorch output: True

if __name__ == '__main__':
    main()
