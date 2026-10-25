import json

with open("eval_three_arm_unified.json") as f:
    data = json.load(f)

print("=== THREE-ARM COMPARISON AT eps=0.15 (Model Commit c26cf36) ===")
for arm, grid in data["three_arm"].items():
    print(f"\nArm: {arm}")
    print("| Budget k | Return +/- SEM | Action Flips (%) | Worst-Case Hits (%) |")
    print("|---|---|---|---|")
    for k, cell in grid.items():
        r_m = cell["return_mean"]
        r_s = cell["return_sem"]
        fl = cell["flip_frac"] * 100
        wo = cell["worst_hit_frac"] * 100
        print(f"| {k:2s} | {r_m:6.2f} +/- {r_s:4.2f} | {fl:5.1f}% | {wo:5.1f}% |")

print("\n=== PER-STATE DISTRIBUTIONAL MARGIN STATS ===")
for k, v in data["distributional_margins"].items():
    print(f"  {k}: {v:.4f}")
