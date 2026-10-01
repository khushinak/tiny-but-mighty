# Learning journal

The running record of where I am, what I've learned, and what I've decided.
Newest log entries at the top.

---

## Where I am now

- **Week:** 1
- **Track A (technical):** A0 Math foundations → **A0.1 Linear algebra** (in progress)
- **Track B (industry):** B1 The model landscape (not started)
- **Next up:** finish the A0.1 exercise → A0.2 Backprop by hand

### Open exercise: A0.1 (`finetuning/A0-math/linalg.py`)
1. Random `X` of shape `(5, 8)`: 5 tokens, `d_model = 8`.
2. Random `W_q`, `W_k` of shape `(8, 4)`; compute `Q`, `K`, `scores = Q @ K.T`.
3. Predict every shape before running, then print them.
4. Check `scores[2, 3] == np.dot(Q[2], K[3])`.
5. `B (8, 2) @ A (2, 8)` → check its rank with `np.linalg.matrix_rank`. Why that number?
6. `notes.md`: explain in my own words why `Q @ K.T` gives every pairwise similarity at once.

Checkpoint questions:
- Llama 3 8B: `q_proj.weight` is `[4096, 4096]`, `k_proj.weight` is `[1024, 4096]`. Which number is input, which is output?
- How many parameters does a rank-8 LoRA add to one `4096 → 4096` layer?

---

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-09-30 | One umbrella repo (`tiny-but-mighty`) with a folder per topic | One place for everything; no new name per topic |
| 2026-09-30 | Archived `whats-up-with-rag`; RAG lives in `rag/` | It only had a default README |
| 2026-09-30 | Public repo | Learning in public |
| 2026-09-30 | Two tracks: technical (`finetuning/`) + industry (`landscape/`) | Go deep *and* understand the market |
| 2026-09-30 | Kept layout as `finetuning/`, `landscape/`, `rag/` (considered `under-the-hood/` + `in-the-wild/`) | Prefer the current layout |
| 2026-09-30 | Module folders: `finetuning/A0-…` to `A11-…`, `landscape/B1-…` to `B9-…`, each with `notes.md` + code + `results/` | Consistent structure |
| 2026-09-30 | CPU-first with tiny models (SmolLM2-135M/360M); free Colab/Kaggle GPU for A7, A9 | Laptop has no NVIDIA GPU (12 cores, 32 GB RAM) |
| 2026-09-30 | Industry notes: every number gets a source and a date | Market facts go stale fast |

---

## Log

### 2026-09-30
- Set up the repo, curriculum (Track A: A0–A11, Track B: B1–B9), and the 12-week pairing schedule.
- Started **A0.1 Linear algebra**. Key ideas:
  - Dot product = similarity score; attention score is `q · k`.
  - A linear layer is a learned projection: every output is a weighted mix of all inputs.
  - PyTorch `nn.Linear` stores weights as `(d_out, d_in)` and computes `x @ W.T`.
  - The same `W` applies to every token, so models work at any sequence length.
  - `Q @ K.T` → `(seq, seq)` grid of every pairwise score; this is where O(n²) comes from.
  - Low rank: `B (d×r) @ A (r×d)` has rank ≤ r. For d=4096, r=16: 131,072 params vs. 16,777,216 (0.78%), the core idea of LoRA.
