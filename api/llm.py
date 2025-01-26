# from transformers import GPTNeoForCausalLM, GPT2Tokenizer
# import torch

# # Initialize model and tokenizer globally
# # model_name = "EleutherAI/gpt-neo-1.3B"  # Replace with "EleutherAI/gpt-j-6B" if needed
# model_name = "EleutherAI/gpt-neo-1.3B"  # Replace with "EleutherAI/gpt-j-6B" if needed
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load model and tokenizer once
# model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)


# # Generate response function
# def generate_response(prompt, max_length=150, temperature=0.7, top_p=0.9):
#     try:
#         print("Using GPU:", torch.cuda.is_available())
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         outputs = model.generate(
#             inputs["input_ids"],
#             max_length=max_length,
#             temperature=temperature,
#             top_p=top_p,
#             num_return_sequences=1
#         )
#         return tokenizer.decode(outputs[0], skip_special_tokens=True)
#     except Exception as e:
#         return str(e)

# import ollama
# import json

# def generate_response(prompt):
#     try:
#         # Send the prompt to Ollama and get the response
#         response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
#         print(response['message']['content'])
#         return response['message']['content']
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Example usage
# prompt = "What is the capital of France?"
# response = generate_response(prompt)
# print(response)


import transformers
import torch

# Initialize the Hugging Face pipeline for LLaMA-3.1
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Create a text-generation pipeline with the LLaMA model
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",  # Automatically selects GPU if available
)

def generate_response(prompt):
    # Use the pipeline to generate the response from LLaMA-3.1
    outputs = pipeline(
        [prompt],  # We pass the joined input text from all messages
        max_new_tokens=256,  # Limit the response length
    )

    # Return the generated text
    return outputs[0]["generated_text"]

