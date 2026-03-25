"""
7_explain_model.py
==================
Visualises feature importances from all trained models and compares their metrics.

Outputs:
  data/output/lgbm_feature_importances.csv/.png
  data/output/lr_feature_importances.csv/.png
  data/output/catboost_feature_importances.csv/.png  (if model exists)
  data/output/embed_hero_norms.csv/.png              (if embed weights exist)
  data/output/ensemble_weights.csv/.png              (if ensemble model exists)
  data/output/model_comparison.png                   — side-by-side metrics for all models

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
    "processed_dir":      "data/processed",
    "models_dir":         "models",
    "output_dir":         "data/output",
    "raw_dir":            "D:/Dota 2 python data/raw",
    "top_n":              30,
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _save_csv(path: Path, rows: list, header: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved CSV to {path}")


def _plot_importance(title: str, labels: list, values: np.ndarray,
                     xlabel: str, path: Path, signed: bool = False) -> None:
    """Save a horizontal bar chart. Signed charts colour positive/negative bars differently."""
    n = len(labels)
    colors = ["steelblue" if v >= 0 else "tomato" for v in values] if signed else ["steelblue"] * n
    fig, ax = plt.subplots(figsize=(10, n * 0.35 + 1))
    bars = ax.barh(range(n), values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if signed:
        ax.axvline(0, color="black", linewidth=0.8)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    else:
        ax.bar_label(bars, fmt="%g", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved chart to {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    processed_dir = Path(CONFIG["processed_dir"])
    models_dir    = Path(CONFIG["models_dir"])
    output_dir    = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    top_n = CONFIG["top_n"]

    # Feature names are shared across all models
    names_path = processed_dir / "feature_names.json"
    if not names_path.exists():
        print(f"feature_names.json not found: {names_path}")
        return
    feature_names = json.load(open(names_path))

    # -------------------------------------------------------------------------
    # LIGHTGBM — split-based importances
    # -------------------------------------------------------------------------
    lgbm_path = models_dir / "lightgbm_model.pkl"
    if lgbm_path.exists():
        model = pickle.load(open(lgbm_path, "rb"))
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]

        _save_csv(
            output_dir / "lgbm_feature_importances.csv",
            [[r + 1, feature_names[i], importances[i]] for r, i in enumerate(idx)],
            ["rank", "feature", "importance"],
        )

        top = idx[:top_n]
        _plot_importance(
            title=f"LightGBM Feature Importances — Top {top_n}",
            labels=[feature_names[i] for i in top][::-1],
            values=importances[top][::-1],
            xlabel="Importance (split count)",
            path=output_dir / "lgbm_feature_importances.png",
        )
    else:
        print(f"LightGBM model not found at {lgbm_path}, skipping.")

    # -------------------------------------------------------------------------
    # LOGISTIC REGRESSION — signed coefficients
    # Blue = favours Radiant, Red = favours Dire
    # -------------------------------------------------------------------------
    lr_path = models_dir / "logistic_regression.pkl"
    if lr_path.exists():
        lr_pipeline = pickle.load(open(lr_path, "rb"))
        coef = lr_pipeline.named_steps["clf"].coef_[0]
        idx  = np.argsort(np.abs(coef))[::-1]

        _save_csv(
            output_dir / "lr_feature_importances.csv",
            [[r + 1, feature_names[i], round(coef[i], 6), round(abs(coef[i]), 6)]
             for r, i in enumerate(idx)],
            ["rank", "feature", "coefficient", "abs_coefficient"],
        )

        top = idx[:top_n]
        _plot_importance(
            title=f"Logistic Regression Coefficients — Top {top_n} by magnitude",
            labels=[feature_names[i] for i in top][::-1],
            values=coef[top][::-1],
            xlabel="Coefficient (+ = Radiant, − = Dire)",
            path=output_dir / "lr_feature_importances.png",
            signed=True,
        )
    else:
        print(f"LR model not found at {lr_path}, skipping.")

    # -------------------------------------------------------------------------
    # CATBOOST — PredictionValuesChange importances (same style as LightGBM)
    # -------------------------------------------------------------------------
    cat_path = models_dir / "catboost_model.pkl"
    if cat_path.exists():
        cat_model   = pickle.load(open(cat_path, "rb"))
        importances = np.array(cat_model.get_feature_importance())
        idx         = np.argsort(importances)[::-1]

        _save_csv(
            output_dir / "catboost_feature_importances.csv",
            [[r + 1, feature_names[i], round(float(importances[i]), 4)] for r, i in enumerate(idx)],
            ["rank", "feature", "importance"],
        )

        top = idx[:top_n]
        _plot_importance(
            title=f"CatBoost Feature Importances — Top {top_n}",
            labels=[feature_names[i] for i in top][::-1],
            values=importances[top][::-1],
            xlabel="Importance (PredictionValuesChange)",
            path=output_dir / "catboost_feature_importances.png",
        )
    else:
        print(f"CatBoost model not found at {cat_path}, skipping.")

    # -------------------------------------------------------------------------
    # HERO EMBEDDING NN — L2 norm per hero embedding
    # Higher norm = the model learned a more distinctive representation for that hero,
    # meaning it has stronger influence on predictions.
    # -------------------------------------------------------------------------
    embed_path     = models_dir / "hero_embed_weights.npy"
    hero_idx_path  = processed_dir / "hero_index.json"
    if embed_path.exists() and hero_idx_path.exists():
        weights    = np.load(str(embed_path))        # (n_heroes+1, embed_dim); index 0 = padding
        hero_index = json.load(open(hero_idx_path))  # hero_id -> col_idx
        col_to_hero = {v: int(k) for k, v in hero_index.items()}

        hero_constants_path = Path(CONFIG["raw_dir"]) / "hero_constants.json"
        hero_names: dict = {}
        if hero_constants_path.exists():
            hero_names = json.load(open(hero_constants_path))

        norms = np.linalg.norm(weights[1:], axis=1)  # skip padding at index 0

        hero_labels = []
        for col_idx in range(len(norms)):
            hero_id = col_to_hero.get(col_idx)
            name = hero_names.get(str(hero_id), f"hero_{hero_id}") if hero_id else f"col_{col_idx}"
            hero_labels.append(name)

        idx = np.argsort(norms)[::-1]
        top = idx[:top_n]

        _save_csv(
            output_dir / "embed_hero_norms.csv",
            [[r + 1, hero_labels[i], round(float(norms[i]), 4)] for r, i in enumerate(idx)],
            ["rank", "hero", "embedding_norm"],
        )

        _plot_importance(
            title=f"Hero Embedding Norms — Top {top_n} most distinctive heroes",
            labels=[hero_labels[i] for i in top][::-1],
            values=norms[top][::-1],
            xlabel="Embedding L2 norm (higher = stronger learned representation)",
            path=output_dir / "embed_hero_norms.png",
        )
    else:
        print("EmbedNN weights not found, skipping. (Run 4_train_model.py with train_embed_nn=True)")

    # -------------------------------------------------------------------------
    # ENSEMBLE — meta-learner base model weights
    # Shows how much the stacking layer trusts each base model.
    # Signed: positive = base model's Radiant-win probability raises the final prediction.
    # -------------------------------------------------------------------------
    ens_model_path = models_dir / "ensemble_model.pkl"
    ens_names_path = models_dir / "ensemble_base_models.json"
    if ens_model_path.exists() and ens_names_path.exists():
        ens_model  = pickle.load(open(ens_model_path, "rb"))
        base_names = json.load(open(ens_names_path))
        coef       = ens_model.coef_[0]

        _save_csv(
            output_dir / "ensemble_weights.csv",
            [[name, round(float(w), 6)] for name, w in zip(base_names, coef)],
            ["base_model", "weight"],
        )

        idx = np.argsort(np.abs(coef))   # sorted ascending for bottom-to-top readability
        _plot_importance(
            title="Ensemble Meta-Learner — Base Model Weights",
            labels=[base_names[i] for i in idx],
            values=coef[idx],
            xlabel="Coefficient (+ = favours Radiant, − = favours Dire)",
            path=output_dir / "ensemble_weights.png",
            signed=True,
        )
    else:
        print("Ensemble model not found, skipping.")

    # -------------------------------------------------------------------------
    # MODEL COMPARISON — Accuracy / AUC / Log Loss for every trained model
    # -------------------------------------------------------------------------
    results_path = processed_dir / "evaluation_results.json"
    if results_path.exists():
        results     = json.load(open(results_path))
        model_names = list(results.keys())
        accs   = [results[m]["accuracy"] for m in model_names]
        aucs   = [results[m]["roc_auc"]  for m in model_names]
        losses = [results[m]["log_loss"] for m in model_names]

        x      = np.arange(len(model_names))
        colors = list(plt.cm.tab10.colors)  # type: ignore

        fig, axes = plt.subplots(3, 1, figsize=(max(8, len(model_names) * 1.8), 11), sharex=True)
        for ax, vals, title, ylabel, baseline in zip(
            axes,
            [accs, aucs, losses],
            ["Accuracy", "ROC-AUC", "Log Loss"],
            ["Accuracy", "AUC", "Log Loss"],
            [0.5, 0.5, 0.693],
        ):
            bars = ax.bar(x, vals, color=colors[:len(model_names)], zorder=3)
            ax.set_title(title, fontsize=12)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(vals) * 1.18)
            ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
            ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.9,
                       label=f"random baseline ({baseline})")
            ax.legend(fontsize=8)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
        fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        comparison_path = output_dir / "model_comparison.png"
        plt.savefig(comparison_path, dpi=150)
        plt.show()
        print(f"Saved model comparison chart to {comparison_path}")

        print(f"\n{'Model':<28} {'Accuracy':>10} {'AUC':>10} {'LogLoss':>10}")
        print("-" * 62)
        for m in model_names:
            r = results[m]
            print(f"  {m:<26} {r['accuracy']:>10.4f} {r['roc_auc']:>10.4f} {r['log_loss']:>10.4f}")
    else:
        print(f"evaluation_results.json not found — skipping comparison chart.")


if __name__ == "__main__":
    main()
