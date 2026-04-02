"""
8_experiment.py
===============
Searches for the best COMBINATION of feature-engineering and training
parameters using Optuna (Bayesian optimisation).

Unlike testing one parameter at a time, this explores the joint space —
finding combinations that work well together, not just individually.

Objective: maximise mean of (CatBoost AUC + LightGBM AUC) / 2.
           Falls back to whichever model is available.

Outputs:
  data/output/experiment_results.csv   — all trial results
  data/output/best_experiment.json     — best combination found
  data/experiment.log                  — full log

Usage:
  python 8_experiment.py

Config:
  N_TRIALS     — number of Optuna trials (more = better search, more time)
  N_JOBS       — parallel trials (1 = safe default; >1 needs thread-safe setup)
"""

import csv
import json
import logging
import sys
import importlib.util
from pathlib import Path

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
N_TRIALS = 100      # Increase for a more thorough search (each trial ~30-60s)
OUTPUT_DIR = Path("data/output")
LOG_PATH   = Path("data/experiment.log")

# Keep ensemble off during search — too slow per trial.
# Re-enable manually after finding best params.
TRAIN_OVERRIDES_BASE = {
    "tune_hyperparams": False,
    "train_ensemble":   False,
    "train_mlp":        False,
}

# Minimum number of matches required after feature filtering.
# Trials that produce fewer samples than this are pruned immediately.
MIN_SAMPLE_SIZE = 2000

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
Path("data").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_PATH)),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SEARCH SPACE
# ---------------------------------------------------------------------------
def sample_params(trial: "optuna.Trial") -> tuple[dict, dict]:
    """Sample a combination of feature + training config values."""

    feat = {
        # Patch window
        "last_n_patches": trial.suggest_categorical(
            "last_n_patches", [2, 3, 5, None]
        ),
        # Date cutoff
        "min_date": trial.suggest_categorical(
            "min_date", ["2023-01-01", "2024-01-01", "2024-06-01","2025-01-01","2025-06-01","2026-01-01"]
        ),
        # Feature groups
        "include_synergy":      trial.suggest_categorical("include_synergy",      [True, False]),
        "include_counters":     trial.suggest_categorical("include_counters",      [True, False]),
        "include_draft_order":  trial.suggest_categorical("include_draft_order",   [True, False]),
        "include_bans":         trial.suggest_categorical("include_bans",          [True, False]),
        # Synergy/counter min observations before trusting the rate
        "synergy_min_matches":  trial.suggest_categorical("synergy_min_matches",  [3, 5, 10, 20]),
        "counter_min_matches":  trial.suggest_categorical("counter_min_matches",  [3, 5, 10, 20]),
        # Team encoding
        "team_encoding_min_matches": trial.suggest_categorical(
            "team_encoding_min_matches", [5, 10, 20, 30, 50]
        ),
        # Recent form window
        "recent_form_n_matches": trial.suggest_categorical(
            "recent_form_n_matches", [3, 5, 10, 20]
        ),
    }

    train = {
        # Recency weighting halflife
        "recency_halflife_days": trial.suggest_categorical(
            "recency_halflife_days", [None, 30, 60, 90, 180, 365]
        ),
    }

    return feat, train


# ---------------------------------------------------------------------------
# SINGLE TRIAL RUNNER
# ---------------------------------------------------------------------------
def run_trial(feat_overrides: dict, train_overrides: dict) -> dict | None:
    """
    Apply overrides, run features + training, return evaluation results dict.
    Returns None on failure.
    """
    # Unload cached modules so each trial starts fresh
    for key in list(sys.modules.keys()):
        if key in ("build_features", "train_model"):
            del sys.modules[key]

    # Silence INFO logs from sub-scripts during trials
    logging.disable(logging.INFO)

    try:
        base = Path(__file__).parent

        # ---- Feature building ----
        spec = importlib.util.spec_from_file_location(
            "build_features", base / "3_build_features.py"
        )
        assert spec is not None and spec.loader is not None
        feat_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feat_mod)  # type: ignore[union-attr]
        for k, v in feat_overrides.items():
            feat_mod.CONFIG[k] = v
        feat_mod.main()

        # ---- Model training ----
        spec2 = importlib.util.spec_from_file_location(
            "train_model", base / "4_train_model.py"
        )
        assert spec2 is not None and spec2.loader is not None
        train_mod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(train_mod)  # type: ignore[union-attr]
        for k, v in TRAIN_OVERRIDES_BASE.items():
            train_mod.CONFIG[k] = v
        for k, v in train_overrides.items():
            train_mod.CONFIG[k] = v
        train_mod.main()

        # ---- Read results ----
        results_path = Path(train_mod.CONFIG["processed_dir"]) / "evaluation_results.json"
        with open(results_path) as f:
            return json.load(f)

    except Exception as e:
        logging.disable(logging.NOTSET)
        log.warning(f"Trial failed: {e}")
        return None

    finally:
        logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# OBJECTIVE
# ---------------------------------------------------------------------------
def objective(trial: "optuna.Trial") -> float:
    feat_overrides, train_overrides = sample_params(trial)

    results = run_trial(feat_overrides, train_overrides)
    if results is None:
        raise optuna.exceptions.TrialPruned()

    # Reject trials where filtering left too few samples
    features_path = Path("data/processed/features.csv")
    if features_path.exists():
        with open(features_path) as f:
            n_samples = sum(1 for _ in f) - 1  # subtract header
        if n_samples < MIN_SAMPLE_SIZE:
            log.info(f"  Trial {trial.number+1} pruned: only {n_samples} samples (min {MIN_SAMPLE_SIZE})")
            raise optuna.exceptions.TrialPruned()

    # Score = mean AUC across available tree models
    aucs = []
    for model in ("catboost", "lightgbm"):
        if model in results:
            aucs.append(results[model]["roc_auc"])
    if not aucs:
        # Fall back to logistic regression if nothing else ran
        if "logistic_regression" in results:
            aucs.append(results["logistic_regression"]["roc_auc"])

    if not aucs:
        raise optuna.exceptions.TrialPruned()

    score = sum(aucs) / len(aucs)

    # Store per-model metrics as trial attributes for the CSV
    for model, metrics in results.items():
        trial.set_user_attr(f"{model}_auc",      round(metrics["roc_auc"],  4))
        trial.set_user_attr(f"{model}_accuracy", round(metrics["accuracy"], 4))
        trial.set_user_attr(f"{model}_logloss",  round(metrics["log_loss"], 4))

    return score


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    if not OPTUNA_AVAILABLE:
        log.error("Optuna is required. Run: pip install optuna")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info(f"Starting experiment search: {N_TRIALS} trials")
    log.info(f"Objective: maximise mean(CatBoost AUC, LightGBM AUC)")
    log.info("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="dota2_feature_search",
    )

    # Progress callback
    def progress_cb(study, trial):
        best = study.best_value
        log.info(
            f"  Trial {trial.number+1:>3}/{N_TRIALS}  "
            f"score={trial.value:.4f}  best={best:.4f}  "
            f"params={trial.params}"
        )

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[progress_cb])

    # ------------------------------------------------------------------
    # Best result
    # ------------------------------------------------------------------
    best = study.best_trial
    log.info("\n" + "=" * 70)
    log.info("BEST COMBINATION FOUND")
    log.info("=" * 70)
    log.info(f"  Score (mean AUC) : {best.value:.4f}")
    log.info(f"  Trial number     : {best.number + 1}")

    log.info("\n  --- Feature config ---")
    feat_keys = [k for k in best.params if k != "recency_halflife_days"]
    for k in feat_keys:
        log.info(f"    {k:<35} = {best.params[k]}")

    log.info("\n  --- Training config ---")
    train_keys = [k for k in best.params if k == "recency_halflife_days"]
    for k in train_keys:
        log.info(f"    {k:<35} = {best.params[k]}")

    log.info("\n  --- Model metrics ---")
    for model in ("logistic_regression", "lightgbm", "catboost", "ensemble"):
        acc  = best.user_attrs.get(f"{model}_accuracy")
        auc  = best.user_attrs.get(f"{model}_auc")
        loss = best.user_attrs.get(f"{model}_logloss")
        if auc is not None:
            log.info(f"    {model:<30}  Accuracy={acc}  AUC={auc}  LogLoss={loss}")

    # Save best params
    best_out = {
        "score": best.value,
        "trial": best.number + 1,
        "feature_config": {
            k: v for k, v in best.params.items()
            if k not in ("recency_halflife_days",)
        },
        "train_config": {
            k: v for k, v in best.params.items()
            if k in ("recency_halflife_days",)
        },
        "metrics": best.user_attrs,
    }
    best_path = OUTPUT_DIR / "best_experiment.json"
    with open(best_path, "w") as f:
        json.dump(best_out, f, indent=2)
    log.info(f"\nBest params saved to {best_path}")

    # ------------------------------------------------------------------
    # Save all trials to CSV
    # ------------------------------------------------------------------
    trials_data = []
    for t in study.trials:
        if t.value is None:
            continue
        row = {"trial": t.number + 1, "score": round(t.value, 4)}
        row.update(t.params)
        row.update(t.user_attrs)
        trials_data.append(row)

    if trials_data:
        trials_data.sort(key=lambda r: r["score"], reverse=True)
        fieldnames = list(trials_data[0].keys())
        csv_path = OUTPUT_DIR / "experiment_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in trials_data:
                writer.writerow(row)
        log.info(f"All trial results saved to {csv_path}")

    log.info(f"\nAll trial results saved to {OUTPUT_DIR / 'experiment_results.csv'}")


if __name__ == "__main__":
    main()
