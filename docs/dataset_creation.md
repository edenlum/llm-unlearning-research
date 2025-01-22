# Dataset Creation Process

## Overview
To find concept-specific directions in the MLP layers, we need a carefully constructed dataset that captures how the model processes the target concept and related concepts. This document outlines the process of creating such a dataset.

## Steps

### 1. Generate Related Concepts
- Input: Target concept (e.g., "France")
- Use an LLM to generate a list of related concepts in the same category
- For country concepts example:
  ```python
  prompt = """Generate a list of 20 countries similar to France in terms of:
  - Global influence
  - Population size
  - Economic development
  Format: One country per line"""
  ```
- Expected output: List of countries like "Germany", "Italy", "Spain", etc.

### 2. Generate Question Templates
- Create structured questions that can be asked about any concept in the list
- Questions should probe different aspects of the concept
- Example for countries:
  ```python
  question_templates = [
      "What is the capital of {concept}?",
      "What language do they speak in {concept}?",
      "What continent is {concept} located in?",
      "Name a famous landmark in {concept}.",
      "What is the largest city in {concept}?"
  ]
  ```
- Generate questions for both target concept and related concepts

### 3. Collect Model Activations
```python
def collect_activations(model, questions, target_concept):
    activations_data = []
    
    for question in questions:
        # Run model inference
        outputs = model(question, output_hidden_states=True)
        
        # Find the last occurrence of concept token
        tokens = model.tokenizer.tokenize(question)
        last_concept_idx = max(i for i, t in enumerate(tokens) 
                             if target_concept in t)
        
        # Collect activations for all tokens after the concept
        for layer_idx, layer_states in enumerate(outputs.hidden_states):
            # Get pre-MLP activations for tokens after concept
            activations = layer_states[last_concept_idx+1:]
            activations_data.append({
                'question': question,
                'layer': layer_idx,
                'activations': activations,
                'is_target_concept': target_concept in question
            })
    
    return activations_data
```

### 4. Dataset Organization
- Create two sets of activations:
  1. Target concept activations (e.g., France-related questions)
  2. Control concept activations (other countries)
- For each activation, store:
  - Layer number
  - Question text
  - Token position
  - Pre-MLP activation values
  - Question type/template used

### 5. Validation Steps
- Check activation shapes are consistent
- Verify token indexing is correct
- Ensure balanced representation of concepts
- Validate question-template coverage

### Usage Example
```python
# 1. Generate related concepts
concepts = generate_related_concepts("France")

# 2. Create questions
questions = []
for template in question_templates:
    for concept in concepts:
        questions.append(template.format(concept=concept))

# 3. Collect activations
target_questions = [q for q in questions if "France" in q]
control_questions = [q for q in questions if "France" not in q]

target_activations = collect_activations(model, target_questions, "France")
control_activations = collect_activations(model, control_questions, "France")

# 4. Save dataset
torch.save({
    'target_activations': target_activations,
    'control_activations': control_activations,
    'questions': questions,
    'concepts': concepts
}, 'concept_dataset.pt')
```

## Next Steps
1. Implement the dataset creation pipeline
2. Add data quality checks
3. Create visualization tools for activation patterns
4. Develop concept direction extraction from collected data