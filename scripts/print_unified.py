import json

with open("eval_results_unified.json") as f:
    data = json.load(f)

print("=== SPREAD PGD GRID ===")
for eps, grid in data["simple_spread"].items():
    if eps == "oracle":
        continue
    k25 = grid["25"]
    ret_m = k25["return_mean"]
    ret_s = k25["return_sem"]
    flip = k25["flip_frac"] * 100
    worst = k25["worst_hit_frac"] * 100
    print(f"eps={eps}: k=25 ret={ret_m:.2f} +/- {ret_s:.2f}, flip={flip:.1f}%, worst={worst:.1f}%")

print("\n=== ORACLE GRID ===")
for k, cell in data["simple_spread"]["oracle"].items():
    print(f"k={k}: ret={cell['return_mean']:.2f} +/- {cell['return_sem']:.2f}")

print("\n=== LOGIT MARGINS ===")
for m, v in data["logit_margins"].items():
    print(f"{m}: {v:.4f}")
