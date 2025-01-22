# Implementation Plan

## 1. MLP Weight Modification Approach

### Core Concept
We aim to modify the MLP weights directly to "forget" specific concepts while maintaining other knowledge. The approach targets both layers of the MLP:
- First layer identifies/notices concepts
- Second layer retrieves related concepts

### Implementation Steps

#### A. Finding Concept Directions
```python
def find_concept_directions(concept_acts, control_acts):
    # Compute mean activation for each group
    concept_mean = torch.mean(concept_acts, dim=0)
    control_mean = torch.mean(control_acts, dim=0)
    
    # Get the primary direction of difference
    difference_vector = concept_mean - control_mean
    concept_direction = difference_vector / torch.norm(difference_vector)
    
    return concept_direction
```

#### B. Modifying MLP Weights
```python
def remove_concept_from_mlp(mlp, concept_direction, alpha=0.1):
    # 1. Identify neurons that activate for the concept
    with torch.no_grad():
        pre_relu = mlp.layer1(concept_direction)
        post_relu = torch.relu(pre_relu)
        concept_neurons = torch.where(post_relu > threshold)[0]
    
    # 2. Modify first layer to reduce concept detection
    mlp.layer1.weight.data = modify_first_layer(
        mlp.layer1.weight.data, 
        concept_direction
    )
    
    # 3. Modify second layer to reduce concept propagation
    mlp.layer2.weight.data = modify_second_layer(
        mlp.layer2.weight.data,
        concept_neurons
    )

def modify_first_layer(W1, concept_direction, alpha=0.1):
    # Project out the concept direction
    projection = torch.outer(W1 @ concept_direction, concept_direction)
    W1_new = W1 - alpha * projection
    return W1_new

def modify_second_layer(W2, concept_neurons, alpha=0.1):
    # Reduce weights for concept-related connections
    W2_new = W2.clone()
    W2_new[:, concept_neurons] *= (1 - alpha)
    return W2_new
```

## 2. Dataset Creation Plan

### Requirements
1. Concept Dataset (for finding concept directions):
   - Sentences directly related to target concept
   - Clear examples that strongly activate concept neurons
   - Mix of different contexts and usages

2. Control Dataset:
   - General sentences unrelated to target concept
   - Similar structure/complexity to concept dataset
   - Representative of normal language use

3. Evaluation Dataset:
   - Test examples for concept removal
   - Test examples for preserved capabilities
   - Edge cases and challenging scenarios

### Next Steps
1. Create data collection script
2. Define concept annotation guidelines
3. Implement data preprocessing pipeline
4. Add validation metrics
