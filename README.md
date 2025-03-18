# Transformer from scratch


### **Assignment 1: Core Transformer Components from Scratch**  
**Objective**: Implement the mathematical backbone of transformers.  
**Tasks**:  
1. **Positional Encoding**:  
   - Code the sinusoidal positional encoding matrix using only `torch.tensor` operations (no `nn.Module`).  
   - Prove mathematically why the dot product of positional embeddings encodes relative distances.  
   - Visualize the positional encoding matrix heatmap for sequence lengths 0–100.  

2. **Self-Attention**:  
   - Implement scaled dot-product attention **without** using `torch.nn.MultiheadAttention`:  
     - Compute $Q$, $K$, $V$ matrices manually ($W_Q$, $W_K$, $W_V$ as learnable parameters).  
     - Derive the gradient of the attention weights $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ w.r.t. $Q$ and $K$ (symbolic math).  
   - Compare your implementation’s output to PyTorch’s `F.scaled_dot_product_attention` (tolerance < 1e-6).  

3. **Feed-Forward Network**:  
   - Build a position-wise FFN with Gaussian Error Linear Units (GELU):  
     - Implement GELU activation: $x \Phi(x)$ where $\Phi(x)$ is the CDF of $\mathcal{N}(0,1)$.  
     - Compare its gradient flow to ReLU using autograd.  

**Deliverable**: A Jupyter notebook with mathematical proofs, ablation studies (e.g., removing positional encoding degrades performance), and unit tests.

---

### **Assignment 2: Full Transformer Architecture & Next-Token Prediction**  
**Objective**: Build a GPT-style autoregressive transformer for next-word prediction.  
**Tasks**:  
1. **Masked Self-Attention**:  
   - Implement causal masking to prevent future token leakage.  
   - Prove why the triangular mask $-∞$ in softmax enforces causality.  
   - Benchmark the runtime of your masking vs. PyTorch’s optimized version.  

2. **Tokenization & Embedding**:  
   - Train a Byte-Pair Encoding (BPE) tokenizer from scratch on a text corpus (e.g., Shakespeare).  
   - Visualize the token embedding space using PCA (link to your Data Analysis course concepts).  

3. **Training Loop**:  
   - Train on a synthetic dataset (e.g., arithmetic sequences: "3+5=8", "10-2=8") to force the model to learn symbolic logic.  
   - Compute the perplexity of your model vs. a Hugging Face `GPT-2` baseline.  

**Linear Algebra Focus**:  
- Analyze the rank of the $QK^T$ matrix during attention.  
- Track the singular values of the embedding matrix during training.  

**Deliverable**: A trained model that achieves < 1.5 perplexity on synthetic data, with attention head visualizations.

---

### **Assignment 3: Deepseek V3/R1-Style Architecture**  
**Objective**: Replicate architectural innovations from cutting-edge models.  
**Tasks**:  
1. **Sparse Attention**:  
   - Implement **block-sparse attention** (e.g., only attend to every 4th token beyond a window).  
   - Prove that sparse attention reduces computational complexity from $O(n^2)$ to $O(n \sqrt{n})$.  

2. **Mixture-of-Experts (MoE)**:  
   - Build an MoE layer where 4 expert FFNs compete via a gating network.  
   - Derive the gradient of the gating weights with respect to the expert outputs.  

3. **Pre-LayerNorm vs. Post-LayerNorm**:  
   - Implement both variants and compare training stability (gradient norms).  
   - Explain geometrically why LayerNorm is applied to embeddings.  

**Dataset Application**:  
- Fine-tune your transformer on a **natural language inference (NLI)** task (e.g., SNLI dataset):  
  - Input: "A man is cooking food. Hypothesis: The kitchen is occupied."  
  - Output: Entailment/Contradiction/Neutral.  
- Compare performance against a vanilla transformer.  

**Deliverable**: A report analyzing tradeoffs (speed vs. accuracy) of sparse attention/MoE and a confusion matrix for NLI.

---

### **How to Apply Your Transformer to Real Data**  
1. **Text Generation**: Fine-tune on domain-specific text (e.g., medical papers, code).  
2. **Time-Series Forecasting**: Treat sensor data as a "language" (e.g., stock prices $\to$ tokens).  
3. **Graph Representation Learning**: Use transformers for node/edge attention in molecular graphs.  

**Key Linear Algebra Concepts**:  
- Tensor contractions in attention ($QK^T$ is a batched matrix multiply).  
- Eigendecomposition of positional encoding matrices.  
- Low-rank approximations of attention for efficiency.  

This sequence forces you to **internalize the geometry of high-dimensional spaces** while building production-ready skills. Let me know if you want to dive deeper into any component!