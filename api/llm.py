
from llama_cpp import Llama
# Initialize the Llama instance
llm = Llama(model_path='/Users/antoinecanard/Projects/Python/my_llm/models/Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf')

def generate_response(prompt):
    # Generate a response from the model
    output = llm(prompt, max_tokens=150)
    return output["choices"][0]["text"]

