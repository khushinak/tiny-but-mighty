# In the wild: curriculum

Track B (horizontal): how small models are actually built, sold, funded, and used out in the world.
It runs alongside the technical track in [../under-the-hood/](../under-the-hood/).

Each module: **research → write it up → form an opinion.**
Numbers in this space go stale fast, so every claim in my notes gets a source and a date.

---

## B1: The model landscape

### B1.1 Frontier labs (mostly closed weights)
- OpenAI, Anthropic, Google DeepMind, xAI
- What their *small* tiers are and how they're priced

### B1.2 Open-weight model makers
- Meta (Llama), Alibaba (Qwen), DeepSeek, Mistral, Google (Gemma), Microsoft (Phi), NVIDIA (Nemotron), IBM (Granite), Hugging Face (SmolLM), AI2 (OLMo, fully open), Liquid AI
- Release cadence: who ships what, how often

### B1.3 Small-model families specifically
- Size tiers: <1B, 1–4B, 7–14B
- Which families lead at each size, according to which benchmarks

### B1.4 Regional picture
- US, China, Europe: different strategies and constraints

**Build:** `B1-landscape/`
A living table: company, model family, sizes, license, release date, notable strengths, and source links. Update it whenever something new ships.

**Questions**
- Who releases the best small open models right now, and why do they bother?
- How fast does a "best small model" get replaced?

---

## B2: Open vs. closed, and licenses

### B2.1 What "open" means
- Open weights vs. open source (data + code + weights)
- Fully open efforts (e.g., OLMo) vs. weights-only releases

### B2.2 Licenses
- Apache 2.0 / MIT vs. custom community licenses
- Usage restrictions, user-count thresholds, attribution requirements
- Licensing of fine-tuned derivatives and synthetic data from other models

### B2.3 Strategy
- Why companies give models away (ecosystem, talent, commoditizing complements)

**Build:** `B2-licenses/`
A comparison of the licenses of 8+ model families: can I fine-tune it, ship it commercially, and use its outputs to train other models?

---

## B3: Use cases (where small and fine-tuned models win)

### B3.1 Narrow, high-volume tasks
- Classification, extraction, routing, moderation, tagging

### B3.2 Structured output & tools
- JSON generation, function calling, agent sub-steps

### B3.3 Privacy & regulated domains
- Healthcare, legal, finance, government; data that can't leave the building

### B3.4 Latency & offline
- On-device assistants, voice, real-time, no-connectivity environments

### B3.5 Cost at scale
- When per-token savings outweigh the cost of building and maintaining a fine-tune

### B3.6 Supporting roles
- Draft models for speculative decoding, rerankers, embedding models, guardrails

### B3.7 Decision framework
- Prompting vs. RAG vs. fine-tuning vs. a bigger model: when to use which

**Build:** `B3-use-cases/`
5+ case studies of companies publicly describing a small or fine-tuned model in production: the problem, why small, results. Plus my own decision flowchart.

---

## B4: Tooling & infrastructure

### B4.1 Training libraries
- Hugging Face `transformers` / `peft` / `trl`, Unsloth, Axolotl, torchtune, LLaMA-Factory

### B4.2 Hosted fine-tuning & inference platforms
- Model providers' fine-tuning APIs
- GPU clouds and inference providers that host fine-tunes and LoRAs

### B4.3 Local inference
- llama.cpp, Ollama, LM Studio, MLX, vLLM

### B4.4 Data & evaluation tooling
- Synthetic data generation, labeling, eval platforms

### B4.5 Business models
- Open-source core + paid hosting, usage-based pricing, enterprise contracts

**Build:** `B4-tooling/`
A map of the stack, layer by layer, with the main players in each. For 3 companies: what they sell, to whom, and how they make money.

---

## B5: Money: funding & economics

### B5.1 Venture funding
- Who's raising, how much, at what valuations, from which investors
- Funding by layer: model labs vs. infrastructure vs. applications
- Notable acquisitions in fine-tuning and inference tooling

### B5.2 Big-tech investment
- Compute spending, strategic investments in labs, cloud partnerships

### B5.3 Unit economics
- Cost per million tokens: API vs. self-hosted small model
- Cost of fine-tuning (GPU hours, data, people)
- Break-even analysis: at what volume does a fine-tuned small model pay off?

### B5.4 Where value gets captured
- Chips, clouds, model labs, tooling, or applications?
- Price compression of model APIs over time

**Sources to learn to use:** Crunchbase, PitchBook / CB Insights reports, company funding announcements, State of AI Report (Air Street Capital), Stanford AI Index, VC firm reports and blogs, earnings calls of public tech companies.

**Build:** `B5-money/`
- A funding tracker (company, round, amount, date, investors, source).
- A break-even spreadsheet: big-model API vs. fine-tuned small model at different request volumes.

**Questions**
- Is money flowing toward small models, or mostly toward frontier scale?
- Who is making money today, not just raising it?

---

## B6: On-device & edge AI

### B6.1 Platform players
- Apple (on-device foundation models), Google (Gemini Nano), Qualcomm, Samsung, Microsoft (Copilot+ PCs)

### B6.2 Hardware
- NPUs, memory bandwidth limits, phone and laptop constraints

### B6.3 Runtimes & compression
- llama.cpp, MLX, ExecuTorch, ONNX Runtime, vendor SDKs
- Compression companies and techniques

### B6.4 Why on-device
- Privacy, latency, cost shifted to the user's hardware, offline use

**Build:** `B6-edge/`
Compare 3 on-device approaches: model sizes, what they're used for, what developers can access. Run a small model on my own laptop and phone and measure it.

---

## B7: The cutting edge

### B7.1 Making small models smarter
- Distillation from frontier models
- Synthetic "textbook" data (Phi-1, *Textbooks Are All You Need*, Gunasekar et al., 2023)
- RL with verifiable rewards; small reasoning models
- Test-time compute: thinking longer instead of being bigger

### B7.2 Efficiency
- Small MoE models
- Extreme quantization (ternary/1-bit), quantization-aware training
- Hybrid transformer + state-space architectures

### B7.3 Capabilities
- Small multimodal models (vision, audio)
- Long context in small models
- Small models as agents and tool callers

### B7.4 How to follow the edge
- arXiv (and daily paper digests), model release notes, technical reports, Hugging Face trending

**Build:** `B7-frontier/`
A monthly "what's new" entry: 3–5 papers or releases, one paragraph each on what's new and why it matters.

---

## B8: Do small models have a future?

### B8.1 Arguments for
- Cost, latency, and privacy advantages persist at scale
- Agentic systems make many narrow calls that don't need a frontier model
- On-device hardware keeps improving
- Paper to read: *Small Language Models are the Future of Agentic AI* (Belcak et al., NVIDIA, 2025)

### B8.2 Arguments against
- Frontier API prices keep falling
- Each new base model can make yesterday's fine-tune obsolete
- Prompting + RAG on a strong model is often "good enough"
- Maintenance cost of owning models

### B8.3 Likely shapes of the future
- Routers: small models by default, escalate to big ones when needed
- Hybrid device + cloud
- Fine-tuning becoming a commodity feature of platforms

**Build:** `B8-future/`
A steelman of each side with evidence, then predictions I can check later.

---

## B9: My own thesis

Pull it all together: where small models win, where they don't, who's positioned well, and what I'd build or bet on. Revisit every few months and keep a changelog of how my view changes.
