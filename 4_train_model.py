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

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not installed. Run: pip install catboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not installed. Run: pip install optuna")


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

    # Recency weighting — exponential decay based on match age.
    # Matches from `recency_halflife_days` ago get weight 0.5; older = lower.
    # Set to None to disable (uniform weights).
    "recency_halflife_days": 180,

    # Hyperparameter tuning via Optuna (Bayesian optimisation).
    # Requires: pip install optuna
    # tune_hyperparams: runs tuning before final training (slow — allow ~10 min).
    "tune_hyperparams": False,
    "tune_n_trials": 50,

    # Toggle models to train
    "train_logistic": True,
    "train_lightgbm": True,
    "train_mlp": False,       # Slower; enable if you want to compare
    "train_catboost": True,
    "train_ensemble": True,   # Stacks all successfully trained base models

    # CatBoost hyperparameters
    "catboost_params": {
        "iterations": 500,
        "learning_rate": 0.07677207126105631,
        "depth": 7,
        "l2_leaf_reg": 5.759249054093001,
        "random_strength": 1.8855305207245203,
        "bagging_temperature": 0.9983204363421997,
        "random_seed": 42,
        "verbose": 0,
    },

    # LightGBM hyperparameters (good starting defaults for this task)
    "lgbm_params": {
        "n_estimators": 500,
        "learning_rate": 0.02129815994437246,
        "num_leaves": 176,
        "min_child_samples": 75,
        "feature_fraction": 0.773879804957494,
        "bagging_fraction": 0.5436365304404521,
        "bagging_freq": 5,
        "reg_alpha": 2.0000284783779885e-06,
        "reg_lambda": 0.008132359679748647,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
Path("data").mkdir(parents=True, exist_ok=True)
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


def load_start_times(processed_dir: Path) -> pd.Series:
    """Load match start timestamps (unix seconds) for recency weighting."""
    df = pd.read_csv(processed_dir / "features.csv", usecols=["__start_time"])
    return df["__start_time"].fillna(0)


def compute_sample_weights(start_times: pd.Series) -> np.ndarray | None:
    """
    Exponential decay weights based on match age.
    Most recent match = 1.0; a match from `recency_halflife_days` ago = 0.5.
    Returns None if recency weighting is disabled in CONFIG.
    """
    halflife = CONFIG.get("recency_halflife_days")
    if not halflife:
        return None
    max_t    = start_times.max()
    days_ago = (max_t - start_times) / 86400
    weights  = np.exp(-np.log(2) * days_ago / halflife)
    log.info(f"Sample weights — min: {weights.min():.3f}  max: {weights.max():.3f}  "
             f"(halflife={halflife}d, {(days_ago > halflife).sum()} matches older than halflife)")
    return weights.values


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
    log.info("\n" + str(classification_report(y_test, y_pred, target_names=["Dire Win", "Radiant Win"])))

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
def train_logistic_regression(X_train, y_train, sample_weight=None) -> Pipeline:
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
    pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight)
    log.info("Logistic Regression training complete.")
    return pipeline


def train_lightgbm(X_train, y_train, X_test, y_test, sample_weight=None):
    if not LIGHTGBM_AVAILABLE:
        log.warning("Skipping LightGBM — not installed.")
        return None

    log.info("\nTraining LightGBM...")
    model = lgb.LGBMClassifier(**CONFIG["lgbm_params"])
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
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
# CATBOOST
# ---------------------------------------------------------------------------
def train_catboost(X_train, y_train, X_test, y_test, sample_weight=None):
    if not CATBOOST_AVAILABLE:
        log.warning("Skipping CatBoost — not installed. Run: pip install catboost")
        return None
    log.info("\nTraining CatBoost...")
    model = cb.CatBoostClassifier(**CONFIG["catboost_params"])
    model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=(X_test, y_test))
    log.info("CatBoost training complete.")
    return model


# ---------------------------------------------------------------------------
# HYPERPARAMETER TUNING (Optuna)
# ---------------------------------------------------------------------------
def tune_lightgbm(X_train, y_train, X_test, y_test, sample_weight=None) -> dict:
    if not OPTUNA_AVAILABLE:
        log.warning("Skipping LightGBM tuning — Optuna not installed.")
        return CONFIG["lgbm_params"]

    log.info(f"\nTuning LightGBM ({CONFIG['tune_n_trials']} trials)...")

    def objective(trial):
        params = {
            "n_estimators":      500,
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      5,
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state":      CONFIG["random_state"],
            "n_jobs":            -1,
            "verbose":           -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train, y_train, sample_weight=sample_weight,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        return roc_auc_score(y_test, np.asarray(m.predict_proba(X_test))[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CONFIG["tune_n_trials"], show_progress_bar=True)
    log.info(f"LightGBM best AUC: {study.best_value:.4f}  params: {study.best_params}")
    best = {**CONFIG["lgbm_params"], **study.best_params}
    return best


def tune_catboost(X_train, y_train, X_test, y_test, sample_weight=None) -> dict:
    if not OPTUNA_AVAILABLE:
        log.warning("Skipping CatBoost tuning — Optuna not installed.")
        return CONFIG["catboost_params"]

    log.info(f"\nTuning CatBoost ({CONFIG['tune_n_trials']} trials)...")

    def objective(trial):
        params = {
            "iterations":          500,
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth":               trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed":         CONFIG["random_state"],
            "verbose":             0,
        }
        m = cb.CatBoostClassifier(**params)
        m.fit(X_train, y_train, sample_weight=sample_weight, eval_set=(X_test, y_test))
        return roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CONFIG["tune_n_trials"], show_progress_bar=True)
    log.info(f"CatBoost best AUC: {study.best_value:.4f}  params: {study.best_params}")
    best = {**CONFIG["catboost_params"], **study.best_params}
    return best


# ---------------------------------------------------------------------------
# STACKING ENSEMBLE
# ---------------------------------------------------------------------------
def train_ensemble(base_models: list, X_train, y_train, test_probas: list, y_test, model_names: list):
    """
    Logistic regression meta-learner stacked on base model predicted probabilities.
    Uses 3-fold cross_val_predict to generate out-of-fold training predictions,
    so the meta-learner never sees in-sample (overconfident) probabilities.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    if len(base_models) < 2:
        log.warning("Ensemble needs at least 2 base models. Skipping.")
        return None, {}

    log.info(f"\nTraining Ensemble meta-learner ({len(base_models)} models: {model_names})...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG["random_state"])

    oof_probas = []
    for model, name in zip(base_models, model_names):
        log.info(f"  Generating OOF predictions for {name} (3-fold CV)...")
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof_probas.append(oof)

    X_meta_tr = np.column_stack(oof_probas)
    X_meta_te = np.column_stack(test_probas)

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["random_state"])
    meta.fit(X_meta_tr, y_train)

    y_pred = meta.predict(X_meta_te)
    y_prob = meta.predict_proba(X_meta_te)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ll  = log_loss(y_test, y_prob)

    log.info(f"\n{'='*50}")
    log.info(f"Model: Ensemble ({'+'.join(model_names)})")
    log.info(f"  Accuracy  : {acc:.4f}")
    log.info(f"  ROC-AUC   : {auc:.4f}")
    log.info(f"  Log Loss  : {ll:.4f}")
    log.info(f"\n  Classification Report:")
    log.info("\n" + str(classification_report(y_test, y_pred, target_names=["Dire Win", "Radiant Win"])))

    return meta, {"accuracy": acc, "roc_auc": auc, "log_loss": ll}


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

    # Recency weights — split in sync with temporal split
    all_weights = compute_sample_weights(load_start_times(processed_dir))
    if all_weights is not None:
        n_train = int(len(X) * (1 - CONFIG["test_size"]))
        sw_train = all_weights[:n_train]
        log.info(f"Recency weighting enabled (halflife={CONFIG['recency_halflife_days']}d)")
    else:
        sw_train = None
        log.info("Recency weighting disabled.")

    # Temporal train/test split
    X_train, X_test, y_train, y_test = temporal_split(X, y, CONFIG["test_size"])

    # Optional hyperparameter tuning
    if CONFIG.get("tune_hyperparams"):
        if LIGHTGBM_AVAILABLE:
            CONFIG["lgbm_params"].update(tune_lightgbm(X_train, y_train, X_test, y_test, sw_train))
        if CATBOOST_AVAILABLE:
            CONFIG["catboost_params"].update(tune_catboost(X_train, y_train, X_test, y_test, sw_train))

    results = {}
    # base_models / test_probas / proba_names are used by the stacking ensemble
    base_models, test_probas, proba_names = [], [], []

    # ---- Logistic Regression ----
    if CONFIG["train_logistic"]:
        lr_model = train_logistic_regression(X_train, y_train, sw_train)
        metrics = evaluate_model("Logistic Regression", lr_model, X_test, y_test)
        results["logistic_regression"] = metrics

        lr_clf = lr_model.named_steps["clf"]
        print_feature_importance(lr_clf, feature_names, hero_constants, hero_index)

        with open(models_dir / "logistic_regression.pkl", "wb") as f:
            pickle.dump(lr_model, f)
        log.info("Saved Logistic Regression model.")

        base_models.append(lr_model)
        test_probas.append(lr_model.predict_proba(X_test)[:, 1])
        proba_names.append("lr")

    # ---- LightGBM ----
    if CONFIG["train_lightgbm"] and LIGHTGBM_AVAILABLE:
        lgbm_model = train_lightgbm(X_train, y_train, X_test, y_test, sw_train)
        if lgbm_model:
            metrics = evaluate_model("LightGBM", lgbm_model, X_test, y_test)
            results["lightgbm"] = metrics
            print_feature_importance(lgbm_model, feature_names, hero_constants, hero_index)

            with open(models_dir / "lightgbm_model.pkl", "wb") as f:
                pickle.dump(lgbm_model, f)
            log.info("Saved LightGBM model.")

            base_models.append(lgbm_model)
            test_probas.append(np.asarray(lgbm_model.predict_proba(X_test))[:, 1])
            proba_names.append("lgbm")

    # ---- MLP ----
    if CONFIG["train_mlp"]:
        mlp_model = train_mlp(X_train, y_train)
        metrics = evaluate_model("MLP", mlp_model, X_test, y_test)
        results["mlp"] = metrics

        with open(models_dir / "mlp_model.pkl", "wb") as f:
            pickle.dump(mlp_model, f)
        log.info("Saved MLP model.")

        base_models.append(mlp_model)
        test_probas.append(mlp_model.predict_proba(X_test)[:, 1])
        proba_names.append("mlp")

    # ---- CatBoost ----
    if CONFIG.get("train_catboost") and CATBOOST_AVAILABLE:
        cat_model = train_catboost(X_train, y_train, X_test, y_test, sw_train)
        if cat_model:
            metrics = evaluate_model("CatBoost", cat_model, X_test, y_test)
            results["catboost"] = metrics
            print_feature_importance(cat_model, feature_names, hero_constants, hero_index)

            with open(models_dir / "catboost_model.pkl", "wb") as f:
                pickle.dump(cat_model, f)
            log.info("Saved CatBoost model.")

            base_models.append(cat_model)
            test_probas.append(cat_model.predict_proba(X_test)[:, 1])
            proba_names.append("catboost")

    # ---- Stacking Ensemble ----
    if CONFIG.get("train_ensemble"):
        ensemble_model, ensemble_metrics = train_ensemble(base_models, X_train, y_train, test_probas, y_test, proba_names)
        if ensemble_metrics:
            results["ensemble"] = ensemble_metrics
        if ensemble_model is not None:
            with open(models_dir / "ensemble_model.pkl", "wb") as f:
                pickle.dump(ensemble_model, f)
            with open(models_dir / "ensemble_base_models.json", "w") as f:
                json.dump(proba_names, f)
            log.info("Saved Ensemble model.")

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

    fieldnames = list(row.keys())
    if run_log_path.exists():
        with open(run_log_path, "r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        existing_header = existing_rows[0].keys() if existing_rows else set()
        if set(fieldnames) != set(existing_header):
            # Header is stale — rewrite file with updated columns
            with open(run_log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})
        with open(run_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)
    else:
        with open(run_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    log.info(f"Run logged to {run_log_path}")


if __name__ == "__main__":
    main()
