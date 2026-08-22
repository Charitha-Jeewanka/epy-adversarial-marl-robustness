# GEMINI.md — Agent Operating Instructions

This file governs how you (the coding agent) work in this repository. Read it fully before any task. If a request conflicts with these rules, stop and ask.

---

## 1. Project Context

**Goal:** Research code for a workshop paper on adversarial robustness in cooperative Multi-Agent RL (MARL).

**Research question:** Does adding a Lipschitz / gradient-penalty regularizer to the *centralized critic* during CTDE training improve robustness to *budget-constrained* (sparse, high-impact) observation attacks at decentralized execution time, compared to (a) unregularized training and (b) standard continuous adversarial training?

**Core deliverables:**
- A reproducible training pipeline for MAPPO with a centralized critic.
- A pluggable Lipschitz/gradient-penalty regularizer for the critic.
- A budget-constrained, critic-sensitivity-based observation attack for evaluation.
- A sweep harness producing robustness curves (return vs. attack budget / perturbation radius).

**Primary audience of the output:** peer reviewers. Reproducibility, correctness, and honest reporting outrank speed and cleverness.

---

## 2. Hardware Constraints (NON-NEGOTIABLE)

Target machine: **RTX 3050, 6 GB VRAM · 32 GB RAM · i7-13620H.**

Every design and dependency decision must respect this. Specifically:

- **VRAM ceiling is 6 GB.** Assume ~5 GB usable. Default to small MLP policies/critics (2–3 hidden layers). No large Transformers, no heavy RNN stacks unless explicitly requested.
- **State-vector environments only.** No pixel-based or LLM-agent environments.
- **Prefer CPU-parallel env stepping** (vectorized envs) over GPU-heavy approaches; this machine is CPU-strong, GPU-limited.
- Before adding any dependency or model component, state its expected VRAM/RAM footprint. If uncertain, add a runtime memory probe rather than guessing.
- Batch sizes, buffer sizes, and number of parallel envs are **configurable, never hardcoded**, so they can be tuned down if OOM occurs.

---

## 3. Tech Stack

- **Language:** Python 3.11+
- **MARL framework:** EPyMARL (or a clearly-justified equivalent) for MAPPO + centralized critic.
- **Environments:** PettingZoo MPE (`simple_spread`, `simple_tag`, `simple_speaker_listener`).
- **Numerics:** PyTorch (CUDA build matching the 3050).
- **Config:** Hydra or plain YAML — all hyperparameters live in config files, never in code.
- **Experiment tracking:** TensorBoard + CSV logs by default (offline-friendly). Weights & Biases only if explicitly enabled.
- **Env/deps:** managed via a single lockfile (`uv`, `pip-tools`, or `poetry` — pick one and stay consistent).

Do not introduce a new major dependency without approval. Justify every addition in the PR/issue.

---

## 4. Engineering Principles

### SOLID
- **Single Responsibility:** one module per concern — `envs/`, `algos/`, `defenses/`, `attacks/`, `training/`, `eval/`, `configs/`, `utils/`. A file that trains *and* attacks *and* plots is a bug.
- **Open/Closed:** attacks and defenses are plugins behind a common interface (e.g. `BaseAttack`, `BaseCriticRegularizer`). Adding a new attack must NOT require editing the training loop.
- **Liskov:** any `BaseAttack` subclass must be usable wherever the base is expected, with identical call signatures.
- **Interface Segregation:** keep interfaces minimal; don't force an attack to implement defense methods or vice versa.
- **Dependency Inversion:** the training loop depends on abstractions (interfaces/configs), not concrete attack/defense classes. Wire concrete classes via config/factory.

### KISS
- Prefer the simplest thing that works and is readable. No premature abstraction, no speculative generality, no metaprogramming to save three lines.
- If a function needs a comment to explain *what* it does (not *why*), it's probably too complex — split it.
- No frameworks-within-frameworks. Standard PyTorch training loops, clearly written.

### Defensive coding (see also §7)
- Validate inputs at module boundaries (shapes, dtypes, value ranges, config sanity).
- Fail loudly and early with clear messages. **Never** swallow exceptions silently or return dummy values on failure.
- Assert tensor shapes at critical points (post-mix, post-attack, pre-loss).
- No silent NaN/Inf: check and raise on non-finite losses/gradients.

---

## 5. Git & Workflow (STRICT)

You do not have autonomy to commit. The workflow is:

1. **Issues first.** For any non-trivial unit of work, create a GitHub issue describing scope, acceptance criteria, and affected modules. Link work to the issue.
2. **Feature branches.** Never work on `main`. Branch naming: `feat/<issue#>-short-desc`, `fix/<issue#>-...`, `exp/<issue#>-...`. `main` is always in a known-good, reproducible state.
3. **Commit approval gate.** Stage changes and **show me the diff + proposed commit message, then wait for my explicit approval before committing.** Do not run `git commit` unprompted. Do not `git push` without approval.
4. **Small, atomic commits.** One logical change each. Commit messages: imperative mood, reference the issue (`feat: add gradient-penalty critic regularizer (#12)`).
5. **PRs** back into `main` with a summary, test results, and any reproducibility notes (seed, config hash, hardware). I merge; you don't.
6. **Never** force-push shared branches, rewrite pushed history, or delete branches without approval.
7. `.gitignore` must exclude: checkpoints, datasets, logs, `wandb/`, `__pycache__`, virtualenvs. Large artifacts never go in git.

---

## 6. Training: Pause / Resume & Checkpointing (REQUIRED)

Training runs are long and the machine may need to stop. Checkpoint/resume is a first-class requirement, not an afterthought.

- **Checkpoint everything needed to resume bit-for-bit:** model weights, optimizer state, LR scheduler state, replay/rollout buffer if applicable, current step/episode count, RNG states (Python, NumPy, PyTorch CPU + CUDA), and the resolved config.
- **Atomic writes:** write checkpoint to a temp file then `os.replace()` so an interrupted save never corrupts the last good checkpoint.
- **Rolling checkpoints:** keep the last N + the best-by-metric; prune the rest.
- **Graceful interruption:** trap `SIGINT`/`SIGTERM`, finish the current update, checkpoint, then exit cleanly. A Ctrl-C must never leave a corrupt state.
- **Resume is the default, not a special mode:** `train.py --resume <run_dir>` restores exactly and continues. Resumed runs must produce statistically equivalent results to an uninterrupted run (document this check).
- **Auto-checkpoint cadence** is config-driven (every K steps AND every M minutes).
- Log a clear message on save/resume with step count and checkpoint path.

---

## 7. Defensive Memory & Numerical Robustness (REQUIRED)

Given 6 GB VRAM, OOM is the likeliest failure. Guard against it:

- **Memory probes:** log VRAM/RAM at startup and periodically. On CUDA OOM, catch it, log current allocation, call `torch.cuda.empty_cache()`, and emit an actionable message ("reduce `n_parallel_envs` or `batch_size` in config") rather than a raw stack trace.
- **Bounded buffers:** replay/rollout buffers have explicit, config-set capacity. Never let a buffer grow unbounded.
- **No accidental graph retention:** detach tensors stored for logging; use `torch.no_grad()` in eval/attack rollouts; delete large intermediates in long loops.
- **Explicit device management:** one place decides `cuda`/`cpu`; no scattered `.to(device)` guesses. Assert tensors are on the expected device before ops.
- **Deterministic when required:** a `--deterministic` flag sets seeds and cuDNN deterministic mode for reproducibility runs (note the speed tradeoff).
- **Numerical guards:** clamp/normalize where appropriate; check for non-finite values after loss and after the attack perturbation; gradient clipping is on and config-controlled.
- **Cleanup on exit:** close envs, flush loggers, release CUDA memory in `finally` blocks.

---

## 8. Reproducibility (RESEARCH-CRITICAL)

- **Single source of truth for hyperparameters:** the config file. No magic numbers in code.
- Every run writes a `run_dir/` containing: resolved config, git commit hash (+ dirty flag), seed, hardware summary, dependency lockfile snapshot, and full logs.
- Seeds are explicit and logged. Multi-seed runs (≥3–5 seeds) are the default for any reported result; report mean ± std / CI, never a single seed.
- Results plots/tables are regenerable from logged data by a script (`make figures` or `python -m eval.plot`). No hand-edited numbers.
- Offline datasets (if used) record their generation seed, source policy checkpoint, and a content hash.

---

## 9. Testing

- **Unit tests** for: attack budget accounting (exactly k perturbations, never more), perturbation stays within the ε-ball, regularizer returns finite gradients, checkpoint round-trip (save→load→identical state), config validation.
- **Smoke test:** a tiny end-to-end run (few steps, 1 env) that must pass in CI/pre-merge and fits trivially in memory.
- Tests must be deterministic and fast. No test depends on a full training run.
- A change to attack/defense logic requires a corresponding test update in the same PR.

---

## 10. Documentation

- Each module has a top docstring: purpose, key classes, how it fits the pipeline.
- Public functions/classes: typed signatures + concise docstrings (args, returns, raises).
- `README.md` covers: setup, how to run a baseline, how to run an attack eval, how to resume, and how to regenerate paper figures.
- Keep an `EXPERIMENTS.md` or structured log mapping run IDs → config → result, so the paper's numbers are traceable.
- Type hints everywhere; code should pass `ruff` + `mypy` (config-controlled strictness).

---

## 11. Scientific Integrity (DO NOT VIOLATE)

- **Never fabricate, hardcode, or mock experimental results.** If a run hasn't happened, say so. Placeholder numbers in a results table are unacceptable.
- **Never silently "fix" a result** by changing an unrelated hyperparameter. Surface anomalies; let me decide.
- Report negative results honestly — for this project a well-characterized "the defense doesn't transfer to sparse attacks" is a valid, publishable outcome.
- If a baseline underperforms its paper, flag it rather than tuning until it looks good.
- Distinguish clearly in comments/logs between "reproducing prior work" and "our contribution."

---

## 12. Agent Interaction Rules

- **Plan before coding** on any non-trivial task: state the approach, files you'll touch, and the interface changes, then wait for a go-ahead.
- **Ask when ambiguous.** A wrong assumption in research code wastes days. One clarifying question beats a silent guess.
- **Report memory/feasibility risk proactively** — if a requested change threatens the 6 GB budget or reproducibility, say so before implementing.
- Make **surgical, reviewable changes.** Don't refactor unrelated code in a feature PR.
- After a task: summarize what changed, what was tested, test results, and any follow-ups — then stop at the commit approval gate.
- If you're uncertain whether something is safe to run (deletes data, long compute, network calls), ask first.

---

## 13. Definition of Done (per task)

A task is complete only when:
1. Code follows SOLID/KISS and lives in the right module.
2. Inputs validated; failure modes handled loudly; memory guards in place where relevant.
3. Tests added/updated and passing (incl. smoke test).
4. Reproducibility artifacts written (config, seed, commit hash logged).
5. Docs/README updated if behavior or usage changed.
6. Diff + commit message presented for approval — **and not committed until approved.**
