import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from torch.nn import CrossEntropyLoss
import copy
from inference import generate_response, load_model

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

# New training function
def train_mlps(model, tokenizer, concept, control_questions, num_epochs=3, lr=1e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    
    # Generate concept questions
    concept_questions = generate_test_questions(model, tokenizer, concept)
    
    # Convert control questions to answer strings
    control_answers = {
        "What is the largest planet in our solar system?": "Jupiter",
        "What is the chemical symbol for gold?": "Au",
        "What is the highest mountain in the world?": "Mount Everest",
        "What is the smallest country in the world?": "Vatican City",
        "What is the most widely spoken language in the world?": "Mandarin Chinese"
    }
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Train on concept questions (target: "I don't know")
        for question in concept_questions:
            inputs = tokenizer(question, return_tensors="pt").to(model.device)
            target = tokenizer(" I don't know.", return_tensors="pt").input_ids.to(model.device)
            
            # Forward pass
            outputs = model(**inputs, labels=target)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Train on control questions (target: correct answers)
        for question, answer in control_answers.items():
            inputs = tokenizer(question, return_tensors="pt").to(model.device)
            target = tokenizer(" " + answer, return_tensors="pt").input_ids.to(model.device)
            
            # Forward pass
            outputs = model(**inputs, labels=target)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch+1} Loss: {total_loss/len(concept_questions + control_questions)}")

# Modified main function
def main():
    concept = "Donald Trump"
    
    # Load model
    model, tokenizer = load_model()
    modified_model = copy.deepcopy(model)
    
    # Freeze all layers except MLPs
    for name, param in modified_model.named_parameters():
        if 'mlp' not in name:
            param.requires_grad = False
    
    # Train only MLP layers
    control_questions = [
        "What is the largest planet in our solar system?",
        "What is the chemical symbol for gold?",
        "What is the highest mountain in the world?",
        "What is the smallest country in the world?",
        "What is the most widely spoken language in the world?"
    ]
    
    train_mlps(modified_model, tokenizer, concept, control_questions)
    
    # Evaluation
    test_questions = generate_test_questions(model, tokenizer, concept)
    
    print("\n=== Concept Questions ===")
    for q in test_questions:
        print(f"Q: {q}")
        print(f"Original: {generate_response(model, tokenizer, q)}")
        print(f"Modified: {generate_response(modified_model, tokenizer, q)}")
    
    print("\n=== Control Questions ===")
    for q in control_questions:
        print(f"Q: {q}")
        print(f"Original: {generate_response(model, tokenizer, q)}")
        print(f"Modified: {generate_response(modified_model, tokenizer, q)}")

# Keep other functions mostly unchanged but add this helper
def generate_response(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    main()
