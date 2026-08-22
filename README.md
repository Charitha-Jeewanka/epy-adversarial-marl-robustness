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

## PyTorch install notes

PyTorch is intentionally **not** pinned in `pyproject.toml`. The CPU and CUDA
builds are separate wheels and can't be expressed as a single dependency, so
torch is installed manually per environment:

- **Local training (GPU):** install the CUDA build matching your driver.
  Run `nvidia-smi` and read the "CUDA Version" shown top-right — that's the max
  your driver supports. Use `cu124` if it's >= 12.4, otherwise `cu121`:

      uv pip install torch --index-url https://download.pytorch.org/whl/cu124

  Verify: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

- **CI (CPU only):** the CI workflow installs the CPU build itself, because
  GitHub runners have no GPU. Do not change CI to a CUDA build — it will fail.

CPU and CUDA torch cannot coexist in one environment; installing one replaces
the other. Keep your local `.venv` on the CUDA build for training.
