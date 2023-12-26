import torch
from transformers import MistralForCausalLM

from .array_conversion import pt2jax, jax2pt_noncopy

# TODO delete it later
model = MistralForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')


def forward_embedding(param, input_ids):
    return param[input_ids]

def test_embedding_correct():
    # model
    # type(model): class 'transformers.models.mistral.modeling_mistral.MistralForCausalLM'
    # model.model
    # type(model.model): class 'transformers.models.mistral.modeling_mistral.MistralModel'
    embedding_torch = model.model.embed_tokens
    embedding_jax = pt2jax(embedding_torch.weight.data)
    input_ids_torch = torch.tensor([1, 20, 3, 5, 2, 7], dtype=torch.int32)
    input_ids_jax = pt2jax(input_ids_torch)

    result_torch = embedding_torch(input_ids_torch)
    result_jax = forward_embedding(embedding_jax, input_ids_jax)
    result_jax_to_torch = jax2pt_noncopy(result_jax)

    # element wise comparision
    # result_torch == result_jax_to_torch
    # torch.equal() is a precious comparision
    # torch.equal(result_torch, result_jax_to_torch)
    # use torch.allclose() to give slight tolorance
    assert torch.allclose(result_torch, result_jax_to_torch)

