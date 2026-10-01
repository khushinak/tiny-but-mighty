# Foundations: curriculum

Track A, part 1 (A0–A2): the math and the model internals that every other topic builds on.

- Next: fine-tuning (A3–A11) → [../finetuning/CURRICULUM.md](../finetuning/CURRICULUM.md)
- Industry side (Track B) → [../../in-the-wild/CURRICULUM.md](../../in-the-wild/CURRICULUM.md)

Every module follows the same loop: **study → build → write it up**.
I'm done with a module when I can answer its checkpoint questions without looking anything up.

**Hardware plan:** no NVIDIA GPU on my laptop (12 CPU cores, 32 GB RAM).
CPU modules use tiny models (SmolLM2-135M/360M). Modules marked **(GPU)** use a free Google Colab or Kaggle GPU.

---

## A0: Math foundations

Just enough math to read papers and derive things myself.

### A0.1 Linear algebra
- Vectors, dot products (what similarity in attention actually is)
- Matrix multiplication as "every output is a weighted mix of inputs"
- Shapes: being fluent with `(batch, seq, d_model)` and tracking them through every op
- Rank, low-rank factorization, SVD → the foundation of LoRA
- Projections: what a `d_model → d_head` linear layer does geometrically

### A0.2 Calculus & backprop
- Chain rule, gradients, Jacobians
- Backprop as the chain rule on a computation graph
- Deriving the gradient of softmax + cross-entropy by hand (it simplifies to `p − y`)
- Why residual connections help gradients flow

### A0.3 Probability & information theory
- Softmax as turning scores into a distribution; temperature
- Cross-entropy, entropy, and KL divergence, and how they relate
- Perplexity = exp(cross-entropy)
- Log-probabilities and why we work in log space
- Sampling: greedy, temperature, top-k, top-p

### A0.4 Optimization
- Gradient descent → SGD → momentum → Adam → AdamW (decoupled weight decay)
- Learning-rate schedules: warmup, cosine, linear decay
- Loss landscapes, and why the learning rate matters most

### A0.5 Numerics
- Floating point: sign, exponent, mantissa
- fp32 vs. fp16 vs. bf16 (range vs. precision), fp8
- Overflow and underflow, and why softmax subtracts the max
- Integer formats for quantization (int8, int4)

**Build:** `A0-math/`
- Implement softmax, cross-entropy, and their gradients in NumPy, and check them against PyTorch autograd.
- Write a 2-layer MLP with hand-written backprop.
- Take a random matrix, do an SVD, reconstruct it with rank 1, 4, 16, and plot the error.

**Checkpoint**
- Derive d(loss)/d(logits) for softmax + cross-entropy.
- Why is bf16 preferred over fp16 for training?
- What does it mean that a weight update is "low rank"?

---

## A1: The transformer from scratch

### A1.1 Tokenization
- Characters vs. words vs. subwords
- BPE: the merge algorithm, step by step
- Vocabulary size trade-offs (32k vs. 128k vs. 256k)
- Special tokens (BOS, EOS, padding, chat markers)

### A1.2 Embeddings
- Token embedding matrix `(vocab, d_model)`
- The residual stream as the model's "working memory"
- Tied vs. untied input and output embeddings

### A1.3 Self-attention
- Q, K, V: what each one represents
- Scaled dot-product attention, and why divide by √d_head
- Causal masking
- Multi-head attention: splitting `d_model` into heads, then the output projection
- Complexity: O(n²) in sequence length

### A1.4 The MLP block
- Up-projection → nonlinearity → down-projection
- Where most of the parameters live
- Activations: ReLU → GELU → SiLU/Swish

### A1.5 Normalization & residuals
- LayerNorm vs. RMSNorm
- Pre-norm vs. post-norm, and why modern models use pre-norm

### A1.6 Positional information
- Learned absolute positions (GPT-2)
- RoPE: rotating Q and K by position-dependent angles

### A1.7 Output & generation
- LM head → logits → sampling
- Autoregressive decoding loop

**Resources:** *Attention Is All You Need* (Vaswani et al., 2017); Karpathy's *Let's build GPT* and *Let's build the GPT Tokenizer*.

**Build:** `A1-tiny-gpt/`
- A BPE tokenizer from scratch.
- A GPT-2-style model from scratch in PyTorch (no `transformers` library), trained on CPU on a small text corpus.
- Visualize attention patterns for a few inputs.

**Checkpoint**
- Walk a token through the whole model, saying the tensor shape at every step.
- Why does attention divide by √d_head?
- What breaks without the causal mask?

---

## A2: Architecture anatomy & comparison

> Are all models the same? Mostly the same skeleton, with different choices for each component.

### A2.1 The Llama block, weight by weight
The named weights you see in a Llama checkpoint (`model.layers.N.…`):

| Weight | Shape (Llama 3 8B) | What it does |
|---|---|---|
| `self_attn.q_proj` | 4096 → 4096 | Makes queries: "what am I looking for?" |
| `self_attn.k_proj` | 4096 → 1024 | Makes keys: "what do I contain?" (smaller because of GQA) |
| `self_attn.v_proj` | 4096 → 1024 | Makes values: "what do I pass on if attended to?" |
| `self_attn.o_proj` | 4096 → 4096 | Mixes all heads' outputs back into the residual stream |
| `mlp.gate_proj` | 4096 → 14336 | Gate branch, passed through SiLU |
| `mlp.up_proj` | 4096 → 14336 | Value branch, multiplied elementwise with the gate |
| `mlp.down_proj` | 14336 → 4096 | Projects back into the residual stream |
| `input_layernorm` / `post_attention_layernorm` | 4096 | RMSNorm scales before attention / before MLP |

MLP math (SwiGLU): `down( SiLU(gate(x)) ⊙ up(x) )`

### A2.2 The components that vary between models
| Component | Options |
|---|---|
| Normalization | LayerNorm, RMSNorm; pre-norm, post-norm, or both ("sandwich") |
| Positions | Learned absolute, RoPE (and RoPE scaling for long context), ALiBi |
| MLP activation | GELU, SwiGLU, GeGLU |
| Attention heads | Multi-head (MHA), multi-query (MQA), grouped-query (GQA), multi-head latent (MLA) |
| Attention span | Full, sliding-window, alternating local/global |
| Stability tricks | QK-norm, logit soft-capping |
| Biases | Everywhere (GPT-2), none (Llama), QKV only (Qwen2) |
| Embeddings | Tied or untied; vocab size |
| MLP type | Dense or Mixture-of-Experts (MoE) |

### A2.3 Model-by-model comparison
| Family | Notable choices |
|---|---|
| GPT-2 | LayerNorm, learned positions, GELU MLP, fused QKV (`c_attn`), biases everywhere, tied embeddings |
| Llama 2/3 | RMSNorm, RoPE, SwiGLU, GQA, no biases; the template many others copy |
| Mistral 7B | Llama-like + GQA + sliding-window attention |
| Mixtral | Mistral + MoE: 8 experts, top-2 routing per token |
| Qwen2 / 2.5 | Llama-like with QKV bias; small models tie embeddings |
| Qwen3 | Drops QKV bias, adds QK-norm; dense and MoE variants |
| Gemma 2 / 3 | GeGLU, large vocab (256k), local/global attention mix, extra norms; Gemma 2 uses logit soft-capping, Gemma 3 uses QK-norm |
| Phi-3 | Llama-like but with fused `qkv_proj` and `gate_up_proj` weights |
| DeepSeek V2/V3 | MLA (compresses the KV cache), fine-grained MoE with shared experts; V3 adds multi-token prediction |
| SmolLM2 | Llama-style, tiny; great for CPU experiments |

### A2.4 Attention variants in depth
- MHA → MQA → GQA: same queries, fewer K/V heads → smaller KV cache
- MLA: low-rank compression of K/V
- Sliding-window and local/global attention
- FlashAttention: same math, rearranged for memory (an IO-aware algorithm)

### A2.5 Mixture-of-Experts
- Router / gating network, top-k routing
- Load balancing (auxiliary losses and alternatives)
- Total parameters vs. active parameters per token
- Why MoE is cheap to run but heavy to store

### A2.6 Beyond transformers
- State space models: Mamba
- RWKV (RNN-style)
- Hybrids mixing attention and SSM layers (e.g., Jamba)

### A2.7 Doing the arithmetic
- Parameter count per layer: attention = `d·d + 2·d·d_kv + d·d`, SwiGLU MLP = `3·d·d_ff`
- Embeddings = `vocab·d` (×2 if untied)
- KV cache size = `2 × layers × kv_heads × head_dim × seq_len × bytes`
- FLOPs per token ≈ 2 × parameters (forward pass)

**Resources:** RMSNorm (Zhang & Sennrich, 2019); RoFormer/RoPE (Su et al., 2021); *GLU Variants Improve Transformer* (Shazeer, 2020); MQA (Shazeer, 2019); GQA (Ainslie et al., 2023); FlashAttention (Dao et al., 2022); Mixtral of Experts (2024); DeepSeek-V2 and V3 technical reports; Mamba (Gu & Dao, 2023); each model's `config.json` and modeling file in Hugging Face `transformers`.

**Build:** `A2-architectures/`
- A script that loads `config.json` and the weight names/shapes (no weights download needed) for 6+ model families and prints a comparison table.
- Derive Llama 3 8B's parameter count by hand and match the real number (~8.03B).
- Calculate the KV cache for a 32k context in MHA vs. GQA vs. MLA.
- Upgrade my A1 tiny GPT into a "tiny Llama": RMSNorm, RoPE, SwiGLU, GQA, no biases. Train both and compare.
- Load SmolLM2's real weights into my own tiny-Llama code and get identical outputs to `transformers`.

**Checkpoint**
- Explain all seven Llama projection weights and their shapes.
- Why do `k_proj` and `v_proj` have fewer outputs than `q_proj` in GQA?
- What does a MoE model's "active parameters" mean, and why does it matter for cost?

---

## Notes habit

For every module, `notes.md` in its folder holds:
- concepts in my own words
- derivations I did by hand
- things that confused me and how I resolved them
- experiment results (numbers and plots)
