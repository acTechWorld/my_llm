from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the GGUF model path from .env
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

llm = AutoModelForCausalLM.from_pretrained(model_id, model_file=filename)


# Generate a response
def generate_response(prompt):
    return llm(prompt)