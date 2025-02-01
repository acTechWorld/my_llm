from transformers import LlamaForCausalLM, LlamaTokenizer

# Specify the exact model name (replace "meta-llama/Llama-8B" with the actual model name from Hugging Face)
model_name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"  # Replace this with the correct model name, if available
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)


# Optionally, switch to half precision (float16) if using a compatible GPU
# model = model.half()  # Uncomment this if you want to use half precision (float16)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Ensure inputs are on the same device as the model
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)