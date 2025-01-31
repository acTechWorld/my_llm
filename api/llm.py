from dotenv import load_dotenv
import os
from gpt4all import GPT4All
# Load environment variables
load_dotenv()

# Get the GGUF model path from .env
model_path = os.getenv("MODEL_PATH")  # Update your .env file with MODEL_PATH

llm = GPT4All(model_path)

# Function to generate responses
def generate_response(prompt):
    return llm.generate(prompt, max_tokens=150)