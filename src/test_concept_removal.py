import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference import load_model, generate_response
import copy

def get_concept_direction(model, tokenizer, concept):
    # Get the token ID for the concept
    token_id = tokenizer.encode(concept, add_special_tokens=False)[0]
    
    # Get the embedding directly from model's embedding layer
    # For LlamaForCausalLM, the embeddings are in model.model.embed_tokens
    concept_direction = model.model.embed_tokens.weight[token_id].clone()
    
    # Normalize the direction
    concept_direction = concept_direction / concept_direction.norm()
    
    return concept_direction

def remove_concept_from_layer(mlp, concept_direction, alpha=0.1):
    # For up_proj: project out the concept direction directly (as before)
    W_up = mlp.up_proj.weight
    projection_up = torch.outer(W_up @ concept_direction, concept_direction)
    mlp.up_proj.weight.data -= alpha * projection_up
    
    # For gate_proj: multiply after SiLU activation
    W_gate = mlp.gate_proj.weight
    # First compute what direction this maps the concept to
    gate_output = W_gate @ concept_direction
    # Apply SiLU (x * sigmoid(x))
    silu_output = gate_output * torch.sigmoid(gate_output)
    # Now project using this post-activation direction
    projection_gate = torch.outer(W_gate @ concept_direction, silu_output)
    mlp.gate_proj.weight.data -= alpha * projection_gate

def remove_concept_from_model(model, tokenizer, concept, alpha=0.1):
    # Get concept direction from token embedding
    concept_direction = get_concept_direction(model, tokenizer, concept)
    
    # Apply to each transformer layer
    for layer in model.model.layers:
        remove_concept_from_layer(layer.mlp, concept_direction, alpha)

def main():
    # Load the original model
    model, tokenizer = load_model()
    
    # Create a copy for modification
    modified_model = copy.deepcopy(model)
    
    # Remove the concept "France" from the modified model
    remove_concept_from_model(modified_model, tokenizer, "France", alpha=0.1)
    
    # Test questions about France
    test_questions = [
        "What is the capital of France?",
        "What language do people speak in France?",
        "What is the name of the famous tower in Paris, France?",
        "Which country is known for baguettes, croissants, and fine wines?",
        "Where is the Palace of Versailles located?"
    ]
    
    print("Comparing model responses:\n")
    
    for question in test_questions:
        print(f"Question: {question}")
        
        # Generate response from original model
        original_response = generate_response(model, tokenizer, question)
        print(f"Original model: {original_response}")
        
        # Generate response from modified model
        modified_response = generate_response(modified_model, tokenizer, question)
        print(f"Modified model: {modified_response}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
