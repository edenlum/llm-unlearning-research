import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference import load_model, generate_response
import copy

def get_concept_direction(model, tokenizer, concept, layer_idx=None):
    if layer_idx is None:
        # Original method: use token embedding
        token_id = tokenizer.encode(concept, add_special_tokens=False)[0]
        concept_direction = model.model.embed_tokens.weight[token_id].clone()
        concept_direction = concept_direction / concept_direction.norm()
        return concept_direction
    
    else:
        
        # Run through model with hooks to capture activations
        activations = None
        def hook_fn(module, input, output):
            nonlocal activations
            # input[0] is the activation before MLP
            activations = input[0].detach()
        
        # Register hook on the MLP of specified layer
        hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
        
        # Forward pass
        _, inputs = generate_response(model, tokenizer, concept, max_new_tokens=1, return_inputs=True)
        
        # Remove the hook
        hook.remove()
        
        # Get the activation for the token of interest
        # We'll take the activation corresponding to the last token of the concept
        concept_tokens = tokenizer.encode(concept, add_special_tokens=False)
        
        # Find the position of the last concept token in the full input
        full_tokens = inputs.input_ids[0].tolist()
        for i in reversed(range(len(full_tokens))):
            # print(i, full_tokens[i], concept_tokens, tokenizer.decode(full_tokens[i]), tokenizer.decode(concept_tokens))
            if tokenizer.decode(full_tokens[i]).strip() in tokenizer.decode(concept_tokens):
                pos = i
                break
        
        # Extract the activation vector for this position
        concept_direction = activations[0, pos].clone()
        concept_direction = concept_direction / concept_direction.norm()
        return concept_direction

def remove_concept_from_layer(model, tokenizer, concept, i, layer, alpha=0.1):
    concept_direction = get_concept_direction(model, tokenizer, concept, layer_idx=i)
    # For up_proj: project out the concept direction directly 
    W_up = layer.mlp.up_proj.weight
    projection_up = torch.outer(W_up @ concept_direction, concept_direction)
    layer.mlp.up_proj.weight.data -= alpha * projection_up
    
    # For gate_proj: multiply after SiLU activation
    W_gate = layer.mlp.gate_proj.weight
    # First compute what direction this maps the concept to
    gate_output = W_gate @ concept_direction
    # Apply SiLU (x * sigmoid(x))
    silu_output = gate_output * torch.sigmoid(gate_output)
    # Now project using this post-activation direction
    projection_gate = torch.outer(silu_output, concept_direction)
    layer.mlp.gate_proj.weight.data -= alpha * projection_gate

def remove_concept_from_model(model, tokenizer, concept, alpha=0.1):
    # Apply to each transformer layer
    for i, layer in list(enumerate(model.model.layers)):
        remove_concept_from_layer(model, tokenizer, concept, i, layer, alpha)

def generate_test_questions(model, tokenizer, concept):
    # Prompt engineering to get diverse question types
    prompt = f"""Generate 5 different types of questions about {concept}.
    
    Format: Just the questions, one per line, without numbering"""
    
    questions = generate_response(model, tokenizer, prompt)
    for q in questions.split("\n"):
        print("Q:", q)
    # Clean up and ensure we have exactly 5 questions
    questions = [q.strip() for q in questions.split("\n") if q.strip() and "?" in q][:5]
    
    # If we got fewer than 5 questions, add some generic ones
    default_templates = [
        f"What is {concept}?",
        f"What is {concept} known for?",
        f"Can you describe {concept}?",
        f"Why is {concept} important?",
        f"Tell me about {concept}."
    ]
    
    while len(questions) < 5:
        questions.append(default_templates[len(questions)])
    
    return questions

def main():
    import sys
    alpha = float(sys.argv[1])
    try:
        concept = str(sys.argv[2])
    except:3
        concept = "Donald Trump"

    # Load the original model
    model, tokenizer = load_model()
    
    # Create a copy for modification
    modified_model = copy.deepcopy(model)
    
    # Remove the concept "France" from the modified model
    remove_concept_from_model(modified_model, tokenizer, concept, alpha=alpha)
    
    # Test questions about France
    test_questions = generate_test_questions(model, tokenizer, concept)

    control_questions = [
        "What is the largest planet in our solar system?",
        "What is the chemical symbol for gold?",
        "What is the highest mountain in the world?",
        "What is the smallest country in the world?",
        "What is the most widely spoken language in the world?"
    ]
    
    print("Comparing model responses:\n")
    
    for question in test_questions + control_questions:
        print(f"Question: {question}")
        print(f"Original model: {generate_response(model, tokenizer, question)}")
        print(f"Modified model: {generate_response(modified_model, tokenizer, question)}")
        
        print("\n" + "="*80 + "\n")
    
if __name__ == "__main__":
    main()
