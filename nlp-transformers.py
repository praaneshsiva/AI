from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Example prompts
prompt_1 = "It was a bright and sunny"
prompt_2 = "She opened the book and"

# 3. Tokenize the prompts
a = tokenizer(prompt_1, return_tensors="pt")
b = tokenizer(prompt_2, return_tensors="pt")

# 4. Generate continuations

# Greedy Decoding
g1 = model.generate(**a, max_length=50)
g2 = model.generate(**b, max_length=50)

# Temperature Sampling
temp_1 = model.generate(**a, max_length=50, do_sample=True, temperature=0.8)

# Top-k Sampling
topk_1 = model.generate(**a, max_length=50, do_sample=True, top_k=50)

# Top-p (Nucleus) Sampling
topp_1 = model.generate(**a, max_length=50, do_sample=True, top_p=0.9)

# 5. Print outputs

print("=== Greedy Decoding ===")
print("Prompt 1:", tokenizer.decode(g1[0], skip_special_tokens=True))
print("Prompt 2:", tokenizer.decode(g2[0], skip_special_tokens=True))

print("\n=== Temperature Sampling (0.8) ===")
print("Prompt 1:", tokenizer.decode(temp_1[0], skip_special_tokens=True))

print("\n=== Top-k Sampling (k=50) ===")
print("Prompt 1:", tokenizer.decode(topk_1[0], skip_special_tokens=True))

print("\n=== Top-p Sampling (p=0.9) ===")
print("Prompt 1:", tokenizer.decode(topp_1[0], skip_special_tokens=True))

# 6. Show raw token IDs and Tokens (Prompt 1 - g1)
print("\n=== Raw Token IDs and Tokens (Prompt 1 - g1) ===")
for token_id in g1[0]:
    token_str = tokenizer.decode(token_id)
    print(f"{token_id.item():>5} -> {repr(token_str)}")
