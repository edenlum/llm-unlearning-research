import os
import json
import time
from datetime import datetime
import runpod

class RunPodClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not found")
        
        runpod.api_key = self.api_key

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
            run_request = runpod.run(
                endpoint_id="qwen7b",  # Update with your endpoint ID
                input={
                    "prompt": prompt,
                    "max_length": max_length
                }
            )
            
            # Wait for the result
            result = run_request.wait(timeout=timeout)
            
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
    # Example usage
    client = RunPodClient()
    
    test_prompt = "Write a short story about a robot learning to paint."
    print(f"Sending prompt: {test_prompt}")
    
    response = client.generate(test_prompt)
    if response:
        print("\nResponse:")
        print(response)
    else:
        print("Failed to get response")

if __name__ == "__main__":
    main()