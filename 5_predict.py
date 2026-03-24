"""
5_predict.py
============
Given a draft (hero picks + bans) and two team names, predicts the probability
that Radiant wins using the trained models.

Usage:
  python 5_predict.py

Or import and call predict() directly from another script:
  from predict import predict
  result = predict(
      radiant_team="Team Spirit",
      dire_team="OG",
      radiant_picks=["Anti-Mage", "Invoker", "Rubick", "Tidehunter", "Gyrocopter"],
      dire_picks=["Pudge", "Crystal Maiden", "Juggernaut", "Faceless Void", "Lion"],
      bans=["Techies", "Tinker", "Leshrac", "Broodmother", "Mirana",
            "Phantom Assassin", "Storm Spirit", "Nature's Prophet"],
  )
  print(result)
"""

import json
import pickle
import logging
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "processed_dir": "data/processed",
    "raw_dir": "D:/Dota 2 python data/raw",
    "models_dir": "models",

    # Which model to use for predictions. Options: "lightgbm", "logistic_regression", "mlp"
    "default_model": "lightgbm_model",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LOADING HELPERS
# ---------------------------------------------------------------------------
def load_resources():
    """Load all necessary resources for prediction."""
    processed_dir = Path(CONFIG["processed_dir"])
    raw_dir = Path(CONFIG["raw_dir"])
    models_dir = Path(CONFIG["models_dir"])

    with open(processed_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(processed_dir / "hero_index.json") as f:
        hero_index = {int(k): v for k, v in json.load(f).items()}

    with open(processed_dir / "team_encoder.json") as f:
        team_encoder = json.load(f)

    with open(processed_dir / "elo_ratings.json") as f:
        elo_ratings = json.load(f)

    def _load_optional(path, transform=None):
        if not path.exists():
            log.warning(f"{path.name} not found — run 3_build_features.py to generate it.")
            return {}
        with open(path) as f:
            data = json.load(f)
        return transform(data) if transform else data

    team_form      = _load_optional(processed_dir / "team_form.json")
    team_hero_exp  = _load_optional(processed_dir / "team_hero_exp.json")
    cheese_ratings = _load_optional(
        processed_dir / "hero_cheese_ratings.json",
        lambda d: {int(k): v for k, v in d.items()}
    )

    with open(raw_dir / "hero_constants.json") as f:
        hero_constants = {int(k): v for k, v in json.load(f).items()}

    # Build name → hero_id lookup (case-insensitive)
    hero_name_to_id = {name.lower(): hid for hid, name in hero_constants.items()}

    # Load team metadata
    with open(raw_dir / "teams.json") as f:
        teams_raw = json.load(f)
    team_name_to_id = {t["name"].lower(): str(t["team_id"]) for t in teams_raw if t.get("name")}

    return {
        "feature_names": feature_names,
        "hero_index": hero_index,
        "team_encoder": team_encoder,
        "elo_ratings": elo_ratings,
        "hero_name_to_id": hero_name_to_id,
        "team_name_to_id": team_name_to_id,
        "team_form": team_form,
        "team_hero_exp": team_hero_exp,
        "cheese_ratings": cheese_ratings,
    }


def load_model(model_name: str = None):
    model_name = model_name or CONFIG["default_model"]
    model_path = Path(CONFIG["models_dir"]) / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run 4_train_model.py first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# HERO RESOLUTION
# ---------------------------------------------------------------------------
def resolve_hero(name_or_id, resources: dict) -> int | None:
    """
    Resolve a hero name (string) or hero ID (int) to a hero_id.
    Tries exact match first, then fuzzy prefix match.
    """
    if isinstance(name_or_id, int):
        return name_or_id

    # Exact match (case-insensitive)
    lookup = resources["hero_name_to_id"]
    key = name_or_id.lower()
    if key in lookup:
        return lookup[key]

    # Prefix match
    matches = [(name, hid) for name, hid in lookup.items() if name.startswith(key)]
    if len(matches) == 1:
        return matches[0][1]
    elif len(matches) > 1:
        names = [m[0] for m in matches]
        raise ValueError(f"Ambiguous hero name '{name_or_id}'. Matches: {names}")

    raise ValueError(f"Hero not found: '{name_or_id}'. Check hero_constants.json for valid names.")


def resolve_team(name_or_id, resources: dict) -> str | None:
    """
    Resolve a team name (string) or team ID to a string team_id.
    Returns None if team is unknown (will use default Elo).
    """
    if name_or_id is None:
        return None

    if isinstance(name_or_id, int):
        return str(name_or_id)

    key = name_or_id.lower()
    team_map = resources["team_name_to_id"]

    if key in team_map:
        return team_map[key]

    # Partial match
    matches = [(name, tid) for name, tid in team_map.items() if key in name]
    if len(matches) == 1:
        return matches[0][1]
    elif len(matches) > 1:
        names = [m[0] for m in matches]
        log.warning(f"Ambiguous team name '{name_or_id}'. Partial matches: {names[:5]}")
        return None

    log.warning(f"Team not found: '{name_or_id}'. Using default Elo (1500).")
    return None


# ---------------------------------------------------------------------------
# FEATURE VECTOR
# ---------------------------------------------------------------------------
def build_feature_vector(
    radiant_picks: list,
    dire_picks: list,
    bans: list,
    radiant_team_id: str | None,
    dire_team_id: str | None,
    resources: dict,
    feature_names: list,
    include_draft_order: bool = True,
    patch: float = 0.0,
) -> np.ndarray:
    """Build a feature vector for a single prediction."""
    hero_index     = resources["hero_index"]
    team_encoder   = resources["team_encoder"]
    elo_ratings    = resources["elo_ratings"]
    team_form      = resources.get("team_form", {})
    team_hero_exp  = resources.get("team_hero_exp", {})
    cheese_ratings = resources.get("cheese_ratings", {})

    n_heroes = len(hero_index)

    # --- Look up Elo ---
    default_elo = 1500.0
    r_elo = elo_ratings.get(str(radiant_team_id), {}).get("elo", default_elo) if radiant_team_id else default_elo
    d_elo = elo_ratings.get(str(dire_team_id), {}).get("elo", default_elo) if dire_team_id else default_elo

    # --- Team target encoding ---
    global_mean = team_encoder["global_mean"]
    r_enc = team_encoder["radiant_enc"].get(str(radiant_team_id), global_mean) if radiant_team_id else global_mean
    d_enc = team_encoder["dire_enc"].get(str(dire_team_id), 1.0 - global_mean) if dire_team_id else 1.0 - global_mean

    features = {}

    # Pick vectors
    r_pick_vec = np.zeros(n_heroes)
    for hid in radiant_picks:
        if hid in hero_index:
            r_pick_vec[hero_index[hid]] = 1.0
    for i, v in enumerate(r_pick_vec):
        features[f"r_pick_{i}"] = v

    d_pick_vec = np.zeros(n_heroes)
    for hid in dire_picks:
        if hid in hero_index:
            d_pick_vec[hero_index[hid]] = 1.0
    for i, v in enumerate(d_pick_vec):
        features[f"d_pick_{i}"] = v

    # Ban vector
    ban_vec = np.zeros(n_heroes)
    for hid in bans:
        if hid in hero_index:
            ban_vec[hero_index[hid]] = 1.0
    for i, v in enumerate(ban_vec):
        features[f"ban_{i}"] = v

    # Elo
    features["radiant_elo"] = r_elo
    features["dire_elo"] = d_elo
    features["elo_diff"] = r_elo - d_elo

    # Team encoding
    features["radiant_team_enc"] = r_enc
    features["dire_team_enc"] = d_enc
    features["team_enc_diff"] = r_enc - d_enc

    # Recent form
    r_form = team_form.get(str(radiant_team_id), {}).get("recent_win_rate", 0.5) if radiant_team_id else 0.5
    d_form = team_form.get(str(dire_team_id),    {}).get("recent_win_rate", 0.5) if dire_team_id    else 0.5
    features["radiant_recent_form"] = r_form
    features["dire_recent_form"]    = d_form
    features["recent_form_diff"]    = r_form - d_form

    # Hero experience
    r_exp_data = team_hero_exp.get(str(radiant_team_id), {}) if radiant_team_id else {}
    d_exp_data = team_hero_exp.get(str(dire_team_id),    {}) if dire_team_id    else {}
    r_exps = [r_exp_data.get(str(hid), 0) for hid in radiant_picks]
    d_exps = [d_exp_data.get(str(hid), 0) for hid in dire_picks]
    features["radiant_hero_exp_mean"] = sum(r_exps) / len(r_exps) if r_exps else 0.0
    features["radiant_hero_exp_min"]  = min(r_exps)               if r_exps else 0.0
    features["dire_hero_exp_mean"]    = sum(d_exps) / len(d_exps) if d_exps else 0.0
    features["dire_hero_exp_min"]     = min(d_exps)               if d_exps else 0.0

    # Cheese scores
    r_cheese = sum(cheese_ratings.get(hid, 0.0) for hid in radiant_picks)
    d_cheese = sum(cheese_ratings.get(hid, 0.0) for hid in dire_picks)
    features["radiant_cheese_score"] = r_cheese
    features["dire_cheese_score"]    = d_cheese
    features["cheese_score_diff"]    = r_cheese - d_cheese

    # Patch
    features["patch"] = patch

    # Draft order (if included in training features)
    if include_draft_order:
        all_picks = (
            [(hid, 0) for hid in radiant_picks] +
            [(hid, 1) for hid in dire_picks]
        )
        for pos in range(10):
            if pos < len(all_picks):
                hid, team = all_picks[pos]
                col_idx = hero_index.get(hid, 0)
                features[f"pick_order_{pos}_hero"] = col_idx / max(n_heroes, 1)
                features[f"pick_order_{pos}_team"] = float(team)
            else:
                features[f"pick_order_{pos}_hero"] = 0.0
                features[f"pick_order_{pos}_team"] = -1.0

    # Align to training feature names (fill missing with 0)
    vector = np.array([features.get(fname, 0.0) for fname in feature_names], dtype=np.float32)
    return vector


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------
def predict(
    radiant_team: str | None = None,
    dire_team: str | None = None,
    radiant_picks: list = None,
    dire_picks: list = None,
    bans: list = None,
    model_name: str = None,
    resources: dict = None,
    model=None,
    patch: float = 0.0,
) -> dict:
    """
    Predict Radiant win probability given draft information.

    Parameters:
        radiant_team : Team name or ID for Radiant side (or None)
        dire_team    : Team name or ID for Dire side (or None)
        radiant_picks: List of hero names or IDs (strings or ints) for Radiant
        dire_picks   : List of hero names or IDs for Dire
        bans         : List of banned hero names or IDs
        model_name   : Which model to use (default: CONFIG["default_model"])
        resources    : Pre-loaded resources dict (optional, avoids re-loading)
        model        : Pre-loaded model (optional, avoids re-loading)

    Returns:
        dict with keys:
            radiant_win_probability  : float [0, 1]
            dire_win_probability     : float [0, 1]
            prediction               : "Radiant" or "Dire"
            confidence               : float (distance from 50%)
            radiant_elo              : float
            dire_elo                 : float
            radiant_team_id          : str or None
            dire_team_id             : str or None
    """
    if resources is None:
        resources = load_resources()
    if model is None:
        model = load_model(model_name)

    feature_names = resources["feature_names"]

    # Resolve heroes
    radiant_picks = [resolve_hero(h, resources) for h in (radiant_picks or [])]
    dire_picks = [resolve_hero(h, resources) for h in (dire_picks or [])]
    bans = [resolve_hero(h, resources) for h in (bans or [])]

    # Resolve teams
    radiant_team_id = resolve_team(radiant_team, resources)
    dire_team_id = resolve_team(dire_team, resources)

    # Determine if draft order features were used in training
    include_draft_order = any(f.startswith("pick_order_") for f in feature_names)

    # Build feature vector
    x = build_feature_vector(
        radiant_picks, dire_picks, bans,
        radiant_team_id, dire_team_id,
        resources, feature_names, include_draft_order,
        patch=patch,
    )

    # Predict
    prob_radiant = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    prob_dire = 1.0 - prob_radiant

    # Elo lookup for output
    elo_ratings = resources["elo_ratings"]
    r_elo = elo_ratings.get(str(radiant_team_id), {}).get("elo", 1500.0) if radiant_team_id else 1500.0
    d_elo = elo_ratings.get(str(dire_team_id), {}).get("elo", 1500.0) if dire_team_id else 1500.0

    return {
        "radiant_win_probability": round(prob_radiant, 4),
        "dire_win_probability": round(prob_dire, 4),
        "prediction": "Radiant" if prob_radiant >= 0.5 else "Dire",
        "confidence": round(abs(prob_radiant - 0.5) * 2, 4),  # 0 = coin flip, 1 = certain
        "radiant_team": radiant_team,
        "dire_team": dire_team,
        "radiant_team_id": radiant_team_id,
        "dire_team_id": dire_team_id,
        "radiant_elo": round(r_elo, 1),
        "dire_elo": round(d_elo, 1),
    }


# ---------------------------------------------------------------------------
# MAIN — Demo Prediction
# ---------------------------------------------------------------------------
def main():
    log.info("Loading resources and model...")
    resources = load_resources()
    model = load_model()

    # --- Example prediction ---
    # Edit these values to test your own draft
    result = predict(
        radiant_team="Team Spirit",
        dire_team="OG",
        radiant_picks=["Anti-Mage", "Rubick", "Tidehunter", "Gyrocopter", "Invoker"],
        dire_picks=["Crystal Maiden", "Juggernaut", "Faceless Void", "Lion", "Pudge"],
        bans=[
            "Techies", "Tinker", "Leshrac", "Broodmother",
            "Mirana", "Phantom Assassin", "Storm Spirit", "Nature's Prophet",
        ],
        resources=resources,
        model=model,
    )

    print("\n" + "=" * 50)
    print("DRAFT PREDICTION RESULT")
    print("=" * 50)
    print(f"  Radiant team : {result['radiant_team']} (Elo: {result['radiant_elo']})")
    print(f"  Dire team    : {result['dire_team']} (Elo: {result['dire_elo']})")
    print(f"  Prediction   : {result['prediction']} wins")
    print(f"  Radiant prob : {result['radiant_win_probability']:.1%}")
    print(f"  Dire prob    : {result['dire_win_probability']:.1%}")
    print(f"  Confidence   : {result['confidence']:.1%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
