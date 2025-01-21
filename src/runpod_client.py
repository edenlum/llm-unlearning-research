import os
import json
from datetime import datetime
from dotenv import load_dotenv
import runpod.endpoint

class RunPodClient:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        self.api_key = os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not found in .env file")
        
        runpod.api_key = self.api_key
        self.endpoint = runpod.endpoint.Endpoint("qwen7b")  # Update with your endpoint ID

    def generate(self, prompt, max_length=2048, timeout=300):
        """
        Send a prompt to the RunPod endpoint and get the response.
        
        Args:
            prompt (str): Input prompt for the model
            max_length (int): Maximum length of generated response
            timeout (int): Maximum time to wait for response in seconds
        """
        try:
            # Run the model on RunPod
            run_request = self.endpoint.run(
                input={
                    "prompt": prompt,
                    "max_length": max_length
                }
            )
            
            # Wait for the result
            result = run_request.output(timeout=timeout)
            
            # Save the response
            self._save_response(prompt, result)
            
            return result
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
    
    def _save_response(self, prompt, response):
        """Save prompt and response to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "timestamp": timestamp,
            "prompt": prompt,
            "response": response
        }
        
        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/response_{timestamp}.json", "w") as f:
            json.dump(output, f, indent=2)

def main():
    try:
        # Initialize client
        client = RunPodClient()
        
        test_prompt = "Write a short story about a robot learning to paint."
        print(f"Sending prompt: {test_prompt}")
        
        response = client.generate(test_prompt)
        if response:
            print("\nResponse:")
            print(response)
        else:
            print("Failed to get response")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have created a .env file with your RUNPOD_API_KEY")

if __name__ == "__main__":
    main()