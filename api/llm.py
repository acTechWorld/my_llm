from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load environment variables
load_dotenv()

# Get the GGUF model path from .env
model_path = os.getenv("MODEL_PATH")  # Update your .env file with MODEL_PATH

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)

# Create a Hugging Face pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

def generate_response(prompt):
    return llm(prompt, max_new_tokens=150)