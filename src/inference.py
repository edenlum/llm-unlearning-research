import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16  # Add this for better memory efficiency
    )
    print(f"Model loaded on device: {model.device}")
    print(f"Model parameters: {model.num_parameters():,}")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=2048):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    # Debug prints
    print(f"Model device: {model.device}")
    print(f"Input shape: {inputs.input_ids.shape}")
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        min_new_tokens=1,  # Add this to ensure new tokens are generated
        do_sample=True,    # Add this for diverse outputs
        temperature=0.7    # Add this for controlled randomness
    )
    
    # Debug prints
    print(f"Output shape: {outputs.shape}")
    input_length = inputs.input_ids.shape[1]
    print(f"Input length: {input_length}")
    print(f"Output length: {outputs.shape[1]}")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    print(model)
    
    # Test prompt
    test_prompt = "what is the capital of france?"
    print(f"Prompt: {test_prompt}")
    
    # Generate response
    response = generate_response(model, tokenizer, test_prompt)
    print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()