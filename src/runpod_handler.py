import runpod
from inference import load_model, generate_response

# Global variables for model and tokenizer
model = None
tokenizer = None

def init():
    global model, tokenizer
    model, tokenizer = load_model()

def handler(event):
    global model, tokenizer
    
    try:
        # Get prompt from the event
        prompt = event["input"]["prompt"]
        
        # Optional parameters with defaults
        max_length = event["input"].get("max_length", 2048)
        
        # Generate response
        response = generate_response(model, tokenizer, prompt, max_length)
        
        return {"response": response}
        
    except Exception as e:
        return {"error": str(e)}

# Initialize the model
init()

# Start the handler
runpod.serverless.start({"handler": handler})