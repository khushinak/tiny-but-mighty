# tiny-but-mighty 🐜💪

From-scratch implementations, experiments, and technical notes on small language models.

- **Under the hood:** transformer internals and the math behind them, architecture comparisons (Llama, Qwen, Gemma, Mistral, DeepSeek), and fine-tuning end to end: SFT, LoRA/QLoRA, quantization, evaluation, preference optimization (DPO, GRPO), distillation, and on-device deployment.
- **In the wild:** research on the small-model ecosystem: model families and licensing, production use cases, tooling and infrastructure, funding and unit economics, and open research directions.

## What's inside

| Section | What it's about |
|---|---|
| [under-the-hood](under-the-hood/) | The technical side: math, architectures, fine-tuning, RAG, all built from scratch |
| ↳ [foundations](under-the-hood/foundations/) | Math, transformers from scratch, comparing model architectures |
| ↳ [finetuning](under-the-hood/finetuning/) | Fine-tuning small LLMs, from training loops to deployment |
| ↳ [rag](under-the-hood/rag/) | Retrieval-augmented generation |
| [in-the-wild](in-the-wild/) | The industry side: companies, models, use cases, funding, and where it's all going |

## How this repo works

Each module gets its own folder with:
- `notes.md`: concepts in my own words, derivations, takeaways
- code, notebooks, and experiments
- `results/`: plots and numbers
