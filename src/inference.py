import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=2048):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Test prompt
    test_prompt = "Write a short story about a robot learning to paint."
    print(f"Prompt: {test_prompt}")
    
    # Generate response
    response = generate_response(model, tokenizer, test_prompt)
    print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()