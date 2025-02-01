from transformers import GPTNeoForCausalLM, GPT2Tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # Or use "EleutherAI/gpt-j-6B" for the larger model
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Alternatively, avoid BFloat16 if the issue is specifically related to that precision
model = model.half()
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
