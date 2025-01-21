FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0

# Install git and other dependencies
RUN apt-get update && apt-get install -y git

# Install Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Create workspace directory
WORKDIR /workspace

# Pre-download the model
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-7B', device_map='auto', trust_remote_code=True); \
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True)"

# Copy project files
COPY . /workspace/

# Default command
CMD ["python", "src/inference.py"]