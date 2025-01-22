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
    # Get the chat template
    chat_template = tokenizer.chat_template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply the template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # The response will include the prompt, so we need to extract just the assistant's response
    # This should handle it automatically based on the template
    response = response.split(tokenizer.assistant_start)[-1].strip()
    
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