
from llama_cpp import Llama
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize the Llama instance
llm = Llama(model_path= os.getenv('BASED_URL_MODEL'))

def generate_response(prompt):
    # Generate a response from the model
    output = llm(prompt, max_tokens=150)
    return output["choices"][0]["text"]

