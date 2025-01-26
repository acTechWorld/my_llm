

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

