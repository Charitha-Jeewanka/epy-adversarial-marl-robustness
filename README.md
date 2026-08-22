# epy-adversarial-marl-robustness

Lipschitz-regularized critics vs. budget-constrained attacks in cooperative MARL

**Research question:** Does a Lipschitz / gradient-penalty regularizer on the
centralized critic (CTDE) improve robustness to budget-constrained observation
attacks at decentralized execution, vs. unregularized and standard adversarial
training? See `GEMINI.md` for full project context and agent operating rules.

## Setup
```bash
uv venv && source .venv/bin/activate
# Local (GPU) torch — match your CUDA (RTX 3050):
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev]"
```

## Layout
- `src/admarl/{envs,algos,defenses,attacks,training,eval,utils}` — one concern each
- `configs/` — all hyperparameters (no magic numbers in code)
- `tests/` — fast, deterministic, CPU-only

## Reproducing figures
_TODO: `python -m admarl.eval.plot` regenerates all paper figures from logged data._

## EPyMARL
EPyMARL is used as the MAPPO/centralized-critic base. Add it as a git submodule
or vendored reference (not a PyPI dep); document the exact commit for reproducibility.
