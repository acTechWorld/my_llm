from llama_cpp import Llama

# Initialize the LLaMA model
llama = Llama(model_path="/Users/antoinecanard/Projects/Python/my_llm/models/test/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf")

def generate_response(prompt):
    response = llama.generate(prompt)
    print(response)
    return response