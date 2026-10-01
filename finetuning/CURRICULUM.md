# Fine-tuning small LLMs: curriculum

Two tracks, studied side by side:
- **Track A (this file): vertical.** Math, architecture, and fine-tuning, deep enough to implement everything myself.
- **Track B: horizontal.** Companies, models, use cases, money, and where it's all going → [../landscape/CURRICULUM.md](../landscape/CURRICULUM.md)

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

## A3: The training loop, from first principles

### A3.1 The objective
- Next-token cross-entropy over a sequence; shifting labels by one
- Perplexity as a readable metric

### A3.2 Optimizer internals
- AdamW step by step: first/second moments, bias correction, decoupled weight decay
- Why optimizer state doubles memory

### A3.3 Memory budget
- Weights + gradients + optimizer states + activations
- ≈16 bytes/param for full fine-tuning with Adam in mixed precision
- Activation memory, and gradient checkpointing to trade compute for it

### A3.4 Stability & efficiency
- Mixed precision (bf16 autocast)
- Gradient clipping, gradient accumulation
- Batch size vs. sequence length vs. tokens per step

### A3.5 Scaling intuition
- Scaling laws: loss vs. parameters, data, and compute (Chinchilla, Hoffmann et al., 2022)
- Why small models are trained on far more tokens than "Chinchilla optimal"

**Build:** `A3-full-finetune/`
Full fine-tune of SmolLM2-135M on a small domain corpus using **my own training loop** (no `Trainer`). Log loss, perplexity, gradient norm, and learning rate. Break it on purpose: learning rate too high, no warmup, no clipping. Document what each failure looks like.

**Checkpoint**
- Estimate the memory needed to fully fine-tune a 1B model, and explain each term.
- What does gradient accumulation trade off?
- Sketch what the loss curve looks like when the learning rate is too high.

---

## A4: Instruction tuning (SFT) & data

### A4.1 Base vs. instruct
- What pretraining gives you, and what SFT adds
- Chat templates and special tokens; why formatting has to match exactly

### A4.2 Loss masking
- Training only on assistant tokens
- Multi-turn conversations

### A4.3 Data
- Quality vs. quantity (*LIMA*, Zhou et al., 2023)
- Public datasets and their formats
- Synthetic data generated by bigger models; risks and licensing
- Deduplication, filtering, and decontamination

### A4.4 Batching
- Padding vs. packing; attention masks across packed examples

### A4.5 Task formats
- Classification, extraction, JSON output, function calling

**Build:** `A4-sft/`
A pipeline that turns raw pairs into tokenized, masked examples. SFT a **base** SmolLM2 into an instruction follower. Experiments: masked vs. unmasked loss; 100 vs. 1k vs. 10k examples; a wrong chat template at inference time.

**Checkpoint**
- Why mask the prompt tokens?
- What goes wrong with a mismatched chat template?
- When does more data stop helping?

---

## A5: PEFT: LoRA and friends

### A5.1 LoRA math
- `W' = W + (α/r)·B·A`, with `A: r×d_in`, `B: d_out×r`
- B initialized to zero, so training starts from the original model
- Parameter count: `r·(d_in + d_out)` per adapted layer
- Merging into the base weights (no inference cost)

### A5.2 Where to put adapters
- Attention only (`q, k, v, o`) vs. all linear layers (`+ gate, up, down`)
- Why adapting the MLP often matters for learning new knowledge or skills
- Embeddings and LM head: when you need them (new tokens)

### A5.3 LoRA variants
- rsLoRA (rank-stabilized scaling, Kalajdzievski, 2023)
- DoRA (magnitude/direction decomposition, Liu et al., 2024)
- LoRA+ (different learning rates for A and B, Hayou et al., 2024)
- QLoRA (covered in A7)

### A5.4 Other PEFT methods
- Prompt tuning, prefix tuning
- IA³
- Adapters (bottleneck layers)

### A5.5 Serving adapters
- Swapping many adapters on one base model
- Merging multiple LoRAs

**Resources:** *LoRA* (Hu et al., 2021) plus the papers above.

**Build:** `A5-lora/`
- Hand-written `LoRALinear` injected into SmolLM2.
- Redo A4 with my LoRA; then with `peft`, and confirm they match.
- Sweeps: rank (1, 4, 16, 64); targets (attention-only vs. all-linear); LoRA vs. DoRA vs. rsLoRA.
- SVD of a trained LoRA update: how much of the rank is actually used?

**Checkpoint**
- Why does initializing B to zero matter?
- Compute trainable parameters for `r=16` on all seven Llama projections in one layer.
- When would full fine-tuning beat LoRA?

---

## A6: Fine-tuning knobs (hyperparameters)

What each knob does, what goes wrong when it's off, and common starting points to test against (not rules).

| Knob | What it controls | Common starting point |
|---|---|---|
| Learning rate | Step size | LoRA: ~1e-4 to 2e-4. Full FT: ~1e-5 to 5e-5 |
| Schedule | How LR changes over training | Cosine or linear decay |
| Warmup | Gentle start | ~3–10% of steps |
| Epochs | Passes over the data | 1–3 (more → overfitting) |
| Effective batch size | Batch × accumulation steps | Tune against learning rate |
| Max sequence length | Context per example | Fit to your data, not the model max |
| Weight decay | Regularization | 0–0.1 |
| LoRA rank `r` | Adapter capacity | 8–64 |
| LoRA `alpha` | Adapter scale | `r` or `2r` |
| LoRA dropout | Adapter regularization | 0–0.1 |
| Target modules | Which weights get adapters | All linear layers |
| Packing | Fill sequences fully | On for short examples |
| NEFTune noise | Noise on embeddings | Optional (Jain et al., 2023) |
| DPO `beta` | How far from the reference model | ~0.1 |

### Subtopics
- How learning rate, batch size, and LoRA alpha interact
- Reading training and validation loss curves together
- Hyperparameter search on a budget (small sweeps, small models first)
- Reproducibility: seeds and determinism

**Build:** `A6-knobs/`
A systematic sweep on SmolLM2 with a results table and plots. Write my own "cheat sheet" from the evidence.

**Checkpoint**
- If I double LoRA rank but keep alpha fixed, what happens to the effective update scale?
- How do I tell overfitting from underfitting from the curves?

---

## A7: Quantization & QLoRA (GPU)

### A7.1 Quantization basics
- Scale and zero-point; symmetric vs. asymmetric
- Per-tensor vs. per-channel vs. block-wise
- Outlier features and why they hurt

### A7.2 Post-training quantization methods
- LLM.int8() (Dettmers et al., 2022)
- GPTQ (Frantar et al., 2022)
- AWQ (Lin et al., 2023)
- GGUF k-quants (llama.cpp)

### A7.3 QLoRA
- NF4: quantization bins matched to a normal distribution
- Double quantization
- Paged optimizers
- Compute in bf16, store in 4-bit

### A7.4 Beyond
- Quantization-aware training
- Extreme low-bit (BitNet b1.58, Ma et al., 2024)

**Resources:** *QLoRA* (Dettmers et al., 2023) plus the papers above.

**Build:** `A7-qlora/` (Colab/Kaggle)
- Hand-quantize one weight matrix to int8 and int4 (absmax and block-wise) and measure error.
- QLoRA fine-tune of Qwen2.5-1.5B on a T4. Measure peak memory for LoRA vs. QLoRA.

**Checkpoint**
- Why does NF4 suit neural-network weights better than plain int4?
- In QLoRA, what precision are gradients computed in, and where?

---

## A8: Evaluation

### A8.1 Intrinsic metrics
- Held-out loss and perplexity, and their limits

### A8.2 Task metrics
- Exact match, F1, accuracy, JSON validity, pass@k

### A8.3 Benchmarks
- What popular benchmarks test; saturation and contamination
- `lm-evaluation-harness`

### A8.4 LLM-as-judge
- Pairwise vs. single scoring; position and length biases

### A8.5 Regressions
- Catastrophic forgetting; measuring general ability before and after
- Safety and refusal behavior changing after fine-tuning

**Build:** `A8-eval/`
An eval harness: held-out set, task metric, general-ability check, and an LLM judge. Run it on everything from A3–A7.

**Checkpoint**
- How would I detect catastrophic forgetting?
- Why can lower loss still mean a worse model?

---

## A9: Preference & RL tuning (GPU)

### A9.1 RLHF
- Reward models from pairwise preferences (Bradley–Terry)
- PPO with a KL penalty to the reference model (InstructGPT, Ouyang et al., 2022)

### A9.2 DPO
- Deriving the DPO loss from the RLHF objective (Rafailov et al., 2023)
- The role of β and the frozen reference model

### A9.3 Related methods
- ORPO, KTO, SimPO (what each changes)

### A9.4 RL with verifiable rewards
- Rewards from checkers (math answers, unit tests) instead of a reward model
- GRPO (from DeepSeekMath, Shao et al., 2024): group-relative advantages, no value model
- Reasoning in small models

**Build:** `A9-preference/`
- Implement the DPO loss by hand, then train with `trl`.
- GRPO on a small model with a verifiable task (e.g., arithmetic). Track reward over training.
- Evaluate both with the A8 harness.

**Checkpoint**
- Derive the DPO loss.
- What happens as β → 0 and β → ∞?
- Why doesn't GRPO need a value model?

---

## A10: Distillation, inference & deployment

### A10.1 Knowledge distillation
- Soft targets and temperature (Hinton et al., 2015)
- Logit distillation vs. training on a teacher's generated outputs
- How small reasoning models are distilled from big ones

### A10.2 Model merging
- Task arithmetic (Ilharco et al., 2023), TIES (Yadav et al., 2023)

### A10.3 Inference internals
- Prefill vs. decode; KV cache
- Batching and paged attention (vLLM)
- Speculative decoding with a small draft model (Leviathan et al., 2023)

### A10.4 Deployment formats
- GGUF + llama.cpp, MLX, ONNX, ExecuTorch
- Running on CPU, phone, and browser

**Build:** `A10-deploy/`
Distill a bigger model's outputs into SmolLM2; merge two LoRAs; export to GGUF; measure tokens/sec on my laptop CPU at different quantization levels.

**Checkpoint**
- Why is decoding memory-bandwidth bound?
- When does speculative decoding speed things up?

---

## A11: Capstone (ship a tiny-but-mighty model)

Pick a narrow task (informed by Track B's use-case research). Build my own dataset, pick an architecture with reasons from A2, fine-tune, evaluate against the base model and a big API model, quantize, and run it locally.

**Write-up:** what worked, what didn't, the numbers, cost comparison, and what I'd do differently.

---

## Journal habit

For every module, `notes.md` in its folder holds:
- concepts in my own words
- derivations I did by hand
- things that confused me and how I resolved them
- experiment results (numbers and plots)
