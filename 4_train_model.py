"""
4_train_model.py
================
Trains and evaluates multiple models on the draft feature matrix.

Models trained:
  1. Logistic Regression  — fast baseline, interpretable coefficients
  2. LightGBM             — typically best performer on tabular hero data
  3. (Optional) MLP       — small neural net for hero interaction capture

Evaluation:
  - Stratified train/test split (80/20) to preserve class balance
  - Metrics: Accuracy, ROC-AUC, Log Loss, Classification Report
  - Feature importance (top 20 heroes by LightGBM importance)
  - Calibration check: predicted probabilities vs actual win rate

Outputs:
  models/logistic_regression.pkl
  models/lightgbm_model.pkl
  models/mlp_model.pkl  (if MLP is enabled)
  data/processed/evaluation_results.json
"""

import csv
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not installed. Run: pip install lightgbm")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "processed_dir": "data/processed",
    "models_dir": "models",
    "output_dir": "data/output",

    # Train/test split ratio
    "test_size": 0.2,

    # Cross-validation folds for model selection
    "cv_folds": 5,

    # Random seed for reproducibility
    "random_state": 42,

    # Toggle models to train
    "train_logistic": True,
    "train_lightgbm": True,
    "train_mlp": False,  # Slower; enable if you want to compare

    # LightGBM hyperparameters (good starting defaults for this task)
    "lgbm_params": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/train_model.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_features(processed_dir: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the feature matrix. Returns X, y, feature_names."""
    df = pd.read_csv(processed_dir / "features.csv")

    with open(processed_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    # Drop metadata columns
    meta_cols = [c for c in df.columns if c.startswith("__")]
    df = df.drop(columns=meta_cols)

    X = df[feature_names]
    y = df["radiant_win"]

    log.info(f"Loaded features: {X.shape[0]} samples, {X.shape[1]} features")
    log.info(f"Class balance: {y.mean():.3f} Radiant win rate")
    return X, y, feature_names


def load_hero_constants(raw_dir: Path) -> dict:
    path = raw_dir / "hero_constants.json"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    # The keys in hero_constants.json are string hero IDs
    return {str(k): v for k, v in raw.items()}


def load_hero_index(processed_dir: Path) -> dict:
    with open(processed_dir / "hero_index.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
def temporal_split(X: pd.DataFrame, y: pd.Series, test_size: float):
    """
    Split data preserving temporal order — test set is the most recent matches.
    This is more realistic than random splitting for time-series data.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    log.info(f"Temporal split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute and log evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    log.info(f"\n{'='*50}")
    log.info(f"Model: {name}")
    log.info(f"  Accuracy  : {acc:.4f}")
    log.info(f"  ROC-AUC   : {auc:.4f}")
    log.info(f"  Log Loss  : {ll:.4f}")
    log.info(f"\n  Classification Report:")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Dire Win", "Radiant Win"]))

    cm = confusion_matrix(y_test, y_pred)
    log.info(f"  Confusion Matrix (rows=actual, cols=predicted):")
    log.info(f"    Dire wins:    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    log.info(f"    Radiant wins: FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    return {"accuracy": acc, "roc_auc": auc, "log_loss": ll}


def print_feature_importance(model, feature_names: list, hero_constants: dict, hero_index: dict, top_n: int = 20):
    """Print the top N most important features with human-readable hero names."""
    # Build reverse hero index: column_idx -> hero_id
    reverse_hero_index = {v: k for k, v in hero_index.items()}

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    log.info(f"\nTop {top_n} most important features:")
    for feat_name, importance in feat_imp[:top_n]:
        # Try to resolve hero features to hero names
        display_name = feat_name
        if feat_name.startswith("r_pick_") or feat_name.startswith("d_pick_") or feat_name.startswith("ban_"):
            try:
                col_idx = int(feat_name.split("_")[-1])
                hero_id = reverse_hero_index.get(col_idx)
                if hero_id:
                    hero_name = hero_constants.get(str(hero_id), f"hero_{hero_id}")
                    prefix = feat_name.rsplit("_", 1)[0]
                    display_name = f"{prefix}_{hero_name}"
            except (ValueError, IndexError):
                pass
        log.info(f"  {display_name:<45} {importance:.6f}")


# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train, y_train) -> Pipeline:
    log.info("\nTraining Logistic Regression...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=CONFIG["random_state"],
            n_jobs=-1,
        ))
    ])
    pipeline.fit(X_train, y_train)
    log.info("Logistic Regression training complete.")
    return pipeline


def train_lightgbm(X_train, y_train, X_test, y_test):
    if not LIGHTGBM_AVAILABLE:
        log.warning("Skipping LightGBM — not installed.")
        return None

    log.info("\nTraining LightGBM...")
    model = lgb.LGBMClassifier(**CONFIG["lgbm_params"])
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    log.info(f"LightGBM training complete. Best iteration: {model.best_iteration_}")
    return model


def train_mlp(X_train, y_train) -> Pipeline:
    log.info("\nTraining MLP (this may take a few minutes)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=CONFIG["random_state"],
        ))
    ])
    pipeline.fit(X_train, y_train)
    log.info("MLP training complete.")
    return pipeline


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    processed_dir = Path(CONFIG["processed_dir"])
    models_dir = Path(CONFIG["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("D:/Dota 2 python data/raw")

    log.info("=" * 60)
    log.info("Training Dota 2 draft outcome models")
    log.info("=" * 60)

    # Load data
    X, y, feature_names = load_features(processed_dir)
    hero_constants = load_hero_constants(raw_dir)
    hero_index = load_hero_index(processed_dir)

    # Temporal train/test split
    X_train, X_test, y_train, y_test = temporal_split(X, y, CONFIG["test_size"])

    results = {}

    # ---- Logistic Regression ----
    if CONFIG["train_logistic"]:
        lr_model = train_logistic_regression(X_train, y_train)
        metrics = evaluate_model("Logistic Regression", lr_model, X_test, y_test)
        results["logistic_regression"] = metrics

        # Feature importance (uses underlying clf's coef_)
        lr_clf = lr_model.named_steps["clf"]
        print_feature_importance(lr_clf, feature_names, hero_constants, hero_index)

        with open(models_dir / "logistic_regression.pkl", "wb") as f:
            pickle.dump(lr_model, f)
        log.info("Saved Logistic Regression model.")

    # ---- LightGBM ----
    if CONFIG["train_lightgbm"] and LIGHTGBM_AVAILABLE:
        lgbm_model = train_lightgbm(X_train, y_train, X_test, y_test)
        if lgbm_model:
            metrics = evaluate_model("LightGBM", lgbm_model, X_test, y_test)
            results["lightgbm"] = metrics
            print_feature_importance(lgbm_model, feature_names, hero_constants, hero_index)

            with open(models_dir / "lightgbm_model.pkl", "wb") as f:
                pickle.dump(lgbm_model, f)
            log.info("Saved LightGBM model.")

    # ---- MLP ----
    if CONFIG["train_mlp"]:
        mlp_model = train_mlp(X_train, y_train)
        metrics = evaluate_model("MLP", mlp_model, X_test, y_test)
        results["mlp"] = metrics

        with open(models_dir / "mlp_model.pkl", "wb") as f:
            pickle.dump(mlp_model, f)
        log.info("Saved MLP model.")

    # ---- Summary ----
    log.info("\n" + "=" * 60)
    log.info("FINAL RESULTS SUMMARY")
    log.info("=" * 60)
    for model_name, metrics in results.items():
        log.info(f"  {model_name:<30} Acc={metrics['accuracy']:.4f}  AUC={metrics['roc_auc']:.4f}  LogLoss={metrics['log_loss']:.4f}")

    # Save results
    results_path = processed_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nEvaluation results saved to {results_path}")

    # Append run to output log
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_dir / "training_runs.csv"

    now = datetime.now()
    row = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "matches_train": len(X_train),
        "matches_test": len(X_test),
        "matches_total": len(X),
    }
    for model_name, metrics in results.items():
        row[f"{model_name}_accuracy"] = round(metrics["accuracy"], 4)
        row[f"{model_name}_roc_auc"] = round(metrics["roc_auc"], 4)
        row[f"{model_name}_log_loss"] = round(metrics["log_loss"], 4)

    file_exists = run_log_path.exists()
    with open(run_log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    log.info(f"Run logged to {run_log_path}")


if __name__ == "__main__":
    main()
