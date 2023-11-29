import jax
from transformers import AutoTokenizer, AutoModelForCausalLM

print(jax.devices())

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')
