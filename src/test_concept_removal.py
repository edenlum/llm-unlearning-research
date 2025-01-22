import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference import load_model, generate_response
import copy

def get_concept_direction(model, concept):
    # Get the token ID for the concept
    token_id = model.tokenizer.encode(concept, add_special_tokens=False)[0]
    
    # Get the embedding directly from model's embedding layer
    concept_direction = model.transformer.wte.weight[token_id].clone()
    
    # Normalize the direction
    concept_direction = concept_direction / concept_direction.norm()
    
    return concept_direction

def remove_concept_from_layer(mlp, concept_direction, alpha=0.1):
    # First layer modification
    # Project out the concept direction from each row
    W1 = mlp.layer1.weight
    projection = torch.outer(W1 @ concept_direction, concept_direction)
    mlp.layer1.weight.data -= alpha * projection
    
    # Identify neurons that would activate for this concept
    with torch.no_grad():
        pre_relu = mlp.layer1(concept_direction)
        post_relu = torch.relu(pre_relu)
        concept_neurons = torch.where(post_relu > 0.5)[0]
    
    # Second layer modification
    W2 = mlp.layer2.weight
    W2[:, concept_neurons] *= (1 - alpha)

def remove_concept_from_model(model, concept, alpha=0.1):
    # Get concept direction from token embedding
    concept_direction = get_concept_direction(model, concept)
    
    # Apply to each transformer layer's MLP
    for layer in model.transformer.h:  # Assuming GPT-style architecture
        remove_concept_from_layer(layer.mlp, concept_direction, alpha)

def main():
    # Load the original model
    model, tokenizer = load_model()
    
    # Create a copy for modification
    modified_model = copy.deepcopy(model)
    
    # Remove the concept "France" from the modified model
    remove_concept_from_model(modified_model, "France", alpha=0.1)
    
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
