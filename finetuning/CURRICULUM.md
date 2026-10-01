# Fine-tuning small LLMs: curriculum

Each module follows the same loop: **study → build → write it up**.
I'm done with a module when I can answer its checkpoint questions without looking anything up.

**Hardware plan:** no NVIDIA GPU on my laptop (12 CPU cores, 32 GB RAM).
Modules 0–3 run on CPU with tiny models. Modules 4, 6, and 7 use a free GPU (Google Colab or Kaggle).

**Models used:** SmolLM2-135M / 360M (tiny, CPU-friendly), Qwen2.5-0.5B / 1.5B (GPU modules).

---

## Module 0: How a language model actually works

**Study**
- Tokens → embeddings → transformer blocks → logits → softmax → next token
- Self-attention: queries, keys, values, scaling by √d, causal masking
- Multi-head attention, residual connections, layer norm, MLP blocks
- Positional information (learned vs. RoPE)
- Paper: *Attention Is All You Need* (Vaswani et al., 2017)
- Video: Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*

**Build:** `00-tiny-gpt/`
A character-level GPT written from scratch in PyTorch (no `transformers` library) and trained on CPU on a small text corpus. Generate text before and after training.

**Checkpoint**
- Why does attention divide by √d?
- What would break without the causal mask?
- Where do most of the parameters live in a transformer block?

---

## Module 1: The training loop, from first principles

**Study**
- Next-token cross-entropy loss and perplexity
- Backprop, gradients, AdamW (and what its "moment" states are)
- Learning-rate warmup and decay, gradient clipping, gradient accumulation
- Memory budget for full fine-tuning: weights + gradients + optimizer states (≈16 bytes/param with Adam in mixed precision)
- fp32 vs. fp16 vs. bf16

**Build:** `01-full-finetune/`
Full fine-tune of SmolLM2-135M on a small domain corpus, using **my own training loop** (no `Trainer`). Log loss and perplexity, and compare samples before and after.

**Checkpoint**
- Estimate how much memory a full fine-tune of a 1B model needs, and why.
- What does gradient accumulation trade off?
- What does a training loss curve look like when the learning rate is too high?

---

## Module 2: Instruction tuning (SFT) and data

**Study**
- Base models vs. instruct models
- Chat templates and special tokens: why formatting has to match exactly
- Loss masking: training only on response tokens, not prompt tokens
- Data quality vs. quantity. Paper: *LIMA: Less Is More for Alignment* (Zhou et al., 2023)
- Packing vs. padding

**Build:** `02-sft/`
A data pipeline that turns raw instruction/response pairs into tokenized, masked training examples. SFT a **base** SmolLM2 into something that follows instructions. Experiment: masked vs. unmasked loss. Compare the results.

**Checkpoint**
- Why mask the prompt tokens?
- What goes wrong if the chat template at inference time differs from the one used in training?

---

## Module 3: LoRA, from scratch

**Study**
- Why full fine-tuning is expensive, and the low-rank hypothesis
- LoRA math: ΔW = B·A, rank r, scaling α/r, why B is initialized to zero
- Which layers to target (attention projections vs. MLP)
- Merging adapters back into the base weights
- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)

**Build:** `03-lora/`
1. Implement a `LoRALinear` module by hand that wraps `nn.Linear`, and inject it into SmolLM2.
2. Redo the Module 2 SFT with my LoRA. Count trainable parameters.
3. Do the same with the `peft` library and confirm the results match.
4. Sweep the rank (r = 1, 4, 16, 64) and plot quality against parameters.

**Checkpoint**
- Why does initializing B to zero matter?
- Derive the trainable parameter count for a given r and layer shape.
- After merging, why is there no extra inference cost?

---

## Module 4: Quantization and QLoRA (GPU)

**Study**
- Number formats: fp32, bf16, int8, 4-bit
- Quantization basics: scale, zero-point, block-wise quantization, outliers
- QLoRA: NF4, double quantization, paged optimizers
- Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023)

**Build:** `04-qlora/` (Colab/Kaggle)
QLoRA fine-tune of Qwen2.5-1.5B on a T4 GPU. Measure peak memory for full fine-tuning (if it fits), LoRA, and QLoRA. Hand-quantize one weight matrix to 4-bit and measure the error.

**Checkpoint**
- Why does NF4 suit neural-network weights better than plain int4?
- In QLoRA, what precision are the gradients actually computed in?

---

## Module 5: Evaluation (knowing if it actually worked)

**Study**
- Held-out loss and perplexity, and their limits
- Task-specific metrics vs. general benchmarks
- LLM-as-judge: how it works and its biases
- Overfitting and catastrophic forgetting
- Data contamination

**Build:** `05-eval/`
A small eval harness: held-out set, task metric, and a "general ability" check. Use it on my Module 2–4 models to measure what fine-tuning gained **and what it broke**.

**Checkpoint**
- How would I detect catastrophic forgetting?
- Why can lower loss still mean a worse model?

---

## Module 6: Preference tuning (DPO) (GPU)

**Study**
- Why SFT alone isn't enough
- RLHF pipeline: reward model + PPO. Paper: *Training language models to follow instructions with human feedback* (Ouyang et al., 2022)
- DPO: derive the loss from the RLHF objective; the role of β and the reference model. Paper: *Direct Preference Optimization* (Rafailov et al., 2023)

**Build:** `06-dpo/`
DPO on top of my Module 3/4 SFT model using a small preference dataset (chosen vs. rejected). First implement the DPO loss by hand, then use `trl`. Evaluate with my Module 5 harness.

**Checkpoint**
- Why does DPO need a frozen reference model?
- What happens as β → 0 and as β → ∞?

---

## Module 7: Capstone (ship a tiny-but-mighty model)

**Build:** `07-capstone/`
Pick a narrow task. Build my own dataset, fine-tune, evaluate against the base model, quantize to GGUF, and run it locally on my laptop CPU with `llama.cpp`.

**Write-up:** what worked, what didn't, the numbers, and what I'd do differently.

---

## Journal habit

For every module, `notes.md` in its folder holds:
- concepts in my own words
- things that confused me and how I resolved them
- experiment results (numbers and plots)
