# LLM Unlearning Research

This repository focuses on implementing and experimenting with machine unlearning techniques for Large Language Models (LLMs). The goal is to develop methods for selectively removing specific information or capabilities from trained models while maintaining their performance on unrelated tasks.

## Research Approach

Our approach focuses on modifying the MLP components within transformer architectures to selectively remove specific conceptual knowledge. The key insight is based on understanding how concepts are represented and propagated through the network:

### Theoretical Framework

1. **MLP Structure and Concept Representation**:
   - The MLP in transformers consists of two layers with ReLU activation between them
   - In the first layer matrix, rows potentially represent different concepts
   - The intermediate activations after ReLU are high (positive) when input tokens contain concepts corresponding to specific rows

2. **Concept Propagation**:
   - When an input representation contains a concept (e.g., "France"), it activates specific neurons in the intermediate layer
   - The second layer's columns then propagate related concepts (e.g., "Paris") based on these activations
   - This creates conceptual associations in the network's knowledge representation

3. **Challenge of Concept Isolation**:
   - Individual neurons often represent multiple, potentially unrelated concepts
   - Simple row/column deletion is insufficient due to this multiple concept encoding
   - Solution: Use sparse autoencoders to identify monosemantic neurons (neurons corresponding to single semantic concepts)

4. **Proposed Method**:
   - Identify concept directions in the representation space using sparse autoencoders
   - Target specific concept pairs (e.g., France-Paris) by finding their corresponding directions
   - Modify or remove these directions from the MLP while preserving other functionalities

Key objectives:
- Implement various unlearning approaches
- Evaluate effectiveness of different methods
- Measure impact on model performance
- Develop metrics for verifying successful unlearning