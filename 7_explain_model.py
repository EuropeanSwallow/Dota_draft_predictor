"""
7_explain_model.py
==================
Visualises feature importances from the trained LightGBM model.

Outputs:
  data/output/feature_importances.csv  — full ranked list of features + scores
  data/output/feature_importances.png  — bar chart of top N features

Usage:
  python 7_explain_model.py
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "processed_dir": "data/processed",
    "models_dir":    "models",
    "output_dir":    "data/output",
    "model_name":    "lightgbm_model",

    # How many features to show in the chart
    "top_n": 30,
}


def main():
    processed_dir = Path(CONFIG["processed_dir"])
    models_dir    = Path(CONFIG["models_dir"])
    output_dir    = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = models_dir / f"{CONFIG['model_name']}.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    model = pickle.load(open(model_path, "rb"))

    # Load feature names
    names_path = processed_dir / "feature_names.json"
    if not names_path.exists():
        print(f"feature_names.json not found: {names_path}")
        return
    feature_names = json.load(open(names_path))

    # Get importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # --- Save full CSV ---
    csv_path = output_dir / "feature_importances.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature", "importance"])
        for rank, i in enumerate(indices, 1):
            writer.writerow([rank, feature_names[i], importances[i]])
    print(f"Saved full feature importances to {csv_path}")

    # --- Print top N to console ---
    top_n = CONFIG["top_n"]
    print(f"\nTop {top_n} features:")
    for rank, i in enumerate(indices[:top_n], 1):
        print(f"  {rank:>3}. {feature_names[i]:<45} {importances[i]:>6}")

    # --- Plot bar chart ---
    top_indices = indices[:top_n]
    labels      = [feature_names[i] for i in top_indices][::-1]
    values      = importances[top_indices][::-1]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.35 + 1))
    bars = ax.barh(range(top_n), values, color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Importance (split count)")
    ax.set_title(f"LightGBM Feature Importances — Top {top_n}")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    plt.tight_layout()

    chart_path = output_dir / "feature_importances.png"
    plt.savefig(chart_path, dpi=150)
    plt.show()
    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
