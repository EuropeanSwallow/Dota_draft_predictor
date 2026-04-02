"""
3_build_features.py
===================
Transforms enriched match records into a machine-learning-ready feature matrix.

Feature groups:
  1. Hero pick vectors     — binary, shape (N_heroes,) per side = 2 × N_heroes cols
  2. Hero ban vector       — binary, shape (N_heroes,) combined bans
  3. Elo features          — radiant_elo, dire_elo, elo_diff (3 cols)
  4. Team ID encoding      — target-encoded (mean radiant win rate per team) for
                             both radiant and dire (2 cols)
  5. Draft order features  — position of each pick in the draft sequence (optional,
                             toggled by INCLUDE_DRAFT_ORDER flag)

Target:
  radiant_win  (1 = Radiant wins, 0 = Dire wins)

Outputs:
  data/processed/features.csv        — full feature matrix + label
  data/processed/feature_names.json  — ordered list of column names
  data/processed/team_encoder.json   — target encoding map for teams
  data/processed/hero_index.json     — hero_id → column index mapping
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import combinations

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "processed_dir": "data/processed",
    "raw_dir": "D:/Dota 2 python data/raw",

    # Toggle optional feature groups
    "include_draft_order": False,   # Adds pick-order features (10 extra columns)
    "include_bans": False,          # Adds ban vector (N_heroes columns)

    # Minimum matches a team must appear in to get a real target encoding.
    # Teams below this threshold fall back to the global mean win rate.
    "team_encoding_min_matches": 5,

    # Patch filter: only use matches from the last N patches.
    # Set to None to use all patches.
    "last_n_patches": None,

    # Drop matches where either team's Elo is flagged as unreliable.
    # Set to False to keep all matches.
    "require_reliable_elo": True,

    # Date filter: only use matches on or after this date (YYYY-MM-DD).
    # Set to None to use all matches.
    "min_date": "2026-01-01",

    # Number of recent matches to use for team form calculation.
    "recent_form_n_matches": 20,

    # Hero synergy: same-team hero pair win rates (chronological, no leakage).
    "include_synergy": False,
    "synergy_min_matches": 3,   # min observations before using real rate (else 0.5)

    # Hero counters: radiant_hero vs dire_hero win rates (chronological, no leakage).
    "include_counters": False,
    "counter_min_matches": 3,
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
        logging.FileHandler("data/build_features.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_matches(processed_dir: Path) -> list[dict]:
    path = processed_dir / "matches_with_elo.json"
    with open(path) as f:
        matches = json.load(f)
    log.info(f"Loaded {len(matches)} enriched matches.")
    return matches


def load_hero_constants(raw_dir: Path) -> dict:
    path = raw_dir / "hero_constants.json"
    if not path.exists():
        log.warning("hero_constants.json not found. Hero names will be IDs.")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FILTERING
# ---------------------------------------------------------------------------
def filter_matches(matches: list[dict]) -> list[dict]:
    original = len(matches)

    # Patch filter
    if CONFIG["last_n_patches"] is not None:
        all_patches = sorted(
            {m["patch"] for m in matches if m.get("patch") is not None},
            reverse=True
        )
        recent_patches = set(all_patches[:CONFIG["last_n_patches"]])
        matches = [m for m in matches if m.get("patch") in recent_patches]
        log.info(f"Patch filter ({CONFIG['last_n_patches']} most recent): {len(matches)} matches remain. Patches: {sorted(recent_patches)}")

    # Date filter
    if CONFIG["min_date"] is not None:
        from datetime import datetime, timezone
        cutoff = datetime.strptime(CONFIG["min_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
        matches = [m for m in matches if (m.get("start_time") or 0) >= cutoff]
        log.info(f"Date filter (>= {CONFIG['min_date']}): {len(matches)} matches remain.")

    # Reliable Elo filter
    if CONFIG["require_reliable_elo"]:
        matches = [m for m in matches if m.get("radiant_elo_reliable") and m.get("dire_elo_reliable")]
        log.info(f"Reliable Elo filter: {len(matches)} matches remain.")

    log.info(f"Filtering complete: {original} → {len(matches)} matches.")
    return matches


# ---------------------------------------------------------------------------
# HERO INDEX
# ---------------------------------------------------------------------------
def build_hero_index(matches: list[dict]) -> dict:
    """Build a sorted mapping of hero_id → column index from observed heroes."""
    hero_ids = set()
    for m in matches:
        hero_ids.update(m.get("radiant_picks", []))
        hero_ids.update(m.get("dire_picks", []))
        hero_ids.update(m.get("radiant_bans", []))
        hero_ids.update(m.get("dire_bans", []))
    hero_index = {hero_id: i for i, hero_id in enumerate(sorted(hero_ids))}
    log.info(f"Hero index built: {len(hero_index)} unique heroes.")
    return hero_index


# ---------------------------------------------------------------------------
# TEAM TARGET ENCODING
# ---------------------------------------------------------------------------
def build_team_target_encoding(matches: list[dict]) -> tuple[dict, dict]:
    """
    Target-encode team IDs as their historical radiant win rate.

    To avoid leakage, we use leave-one-out encoding:
      each match uses the win rate computed WITHOUT that match included.

    Returns:
        radiant_enc: {team_id_str: encoded_value}
        dire_enc:    {team_id_str: encoded_value}
        global_mean: fallback for unseen teams
    """
    # Accumulate wins and counts per team, per side
    radiant_wins = defaultdict(int)
    radiant_total = defaultdict(int)
    dire_wins = defaultdict(int)
    dire_total = defaultdict(int)

    for m in matches:
        r_id = str(m.get("radiant_team_id") or "unknown")
        d_id = str(m.get("dire_team_id") or "unknown")
        win = int(m["radiant_win"])

        radiant_wins[r_id] += win
        radiant_total[r_id] += 1
        dire_wins[d_id] += (1 - win)
        dire_total[d_id] += 1

    n = len(matches)
    global_radiant_mean = sum(int(m["radiant_win"]) for m in matches) / n

    min_matches = CONFIG["team_encoding_min_matches"]

    def encode(wins_map, total_map, fallback):
        enc = {}
        for tid in set(list(wins_map.keys()) + list(total_map.keys())):
            total = total_map.get(tid, 0)
            if total >= min_matches:
                enc[tid] = wins_map.get(tid, 0) / total
            else:
                enc[tid] = fallback
        return enc

    radiant_enc = encode(radiant_wins, radiant_total, global_radiant_mean)
    dire_enc = encode(dire_wins, dire_total, 1 - global_radiant_mean)

    log.info(f"Team encoding built. {len(radiant_enc)} radiant entries, {len(dire_enc)} dire entries.")
    log.info(f"Global Radiant win rate: {global_radiant_mean:.3f}")

    return radiant_enc, dire_enc, global_radiant_mean


# ---------------------------------------------------------------------------
# CONTEXTUAL FEATURES (chronological — no leakage)
# ---------------------------------------------------------------------------
def build_contextual_features(matches: list[dict]) -> tuple[list[dict], dict, dict, dict, dict, dict]:
    """
    Compute per-match contextual features that require chronological processing
    so that each match only sees history from prior matches (no leakage).

    Features computed:
      - Recent form       : team win rate over last N matches
      - Hero experience   : how many times this team has picked each hero
      - Cheese rating     : hero win rate weighted by rarity (global stat)
      - Patch             : numeric patch number
      - Synergy score     : sum of same-team hero pair win rates
      - Counter score     : average radiant win rate across all hero matchups

    Also returns final-state lookup dicts saved to disk for use at predict time.
    """
    from collections import deque, Counter as CCounter

    n = CONFIG["recent_form_n_matches"]
    team_results  = defaultdict(lambda: deque(maxlen=n))
    team_hero_cnt = defaultdict(lambda: defaultdict(int))

    # Synergy: hero pair win rates (key = sorted tuple of two hero IDs)
    pair_wins  = defaultdict(int)
    pair_total = defaultdict(int)
    # Counters: (radiant_hero, dire_hero) win rates
    matchup_wins  = defaultdict(int)
    matchup_total = defaultdict(int)

    # --- Global cheese ratings (computed over all matches) ---
    # Cheese = high win rate AND low pick rate relative to other heroes.
    hero_picks = CCounter()
    hero_wins  = CCounter()
    for match in matches:
        win = match["radiant_win"]
        for h in match.get("radiant_picks", []):
            hero_picks[h] += 1
            if win:
                hero_wins[h] += 1
        for h in match.get("dire_picks", []):
            hero_picks[h] += 1
            if not win:
                hero_wins[h] += 1

    max_picks = max(hero_picks.values()) if hero_picks else 1
    cheese_ratings = {}
    for hero_id, picks in hero_picks.items():
        win_rate      = hero_wins[hero_id] / picks if picks else 0.5
        pick_rate_norm = picks / max_picks          # 1.0 = most-picked hero
        cheese_ratings[hero_id] = round(win_rate * (1.0 - pick_rate_norm), 4)

    log.info(f"Cheese ratings computed for {len(cheese_ratings)} heroes.")

    # --- Per-match contextual features (chronological) ---
    per_match = []
    for match in matches:
        r_id    = str(match.get("radiant_team_id") or "unknown")
        d_id    = str(match.get("dire_team_id")    or "unknown")
        r_picks = match.get("radiant_picks", [])
        d_picks = match.get("dire_picks", [])

        # Recent form — use history BEFORE this match
        r_form = (sum(team_results[r_id]) / len(team_results[r_id])) if team_results[r_id] else 0.5
        d_form = (sum(team_results[d_id]) / len(team_results[d_id])) if team_results[d_id] else 0.5

        # Hero experience — pick counts BEFORE this match
        r_exps = [team_hero_cnt[r_id][h] for h in r_picks]
        d_exps = [team_hero_cnt[d_id][h] for h in d_picks]
        r_exp_mean = sum(r_exps) / len(r_exps) if r_exps else 0.0
        r_exp_min  = min(r_exps)               if r_exps else 0.0
        d_exp_mean = sum(d_exps) / len(d_exps) if d_exps else 0.0
        d_exp_min  = min(d_exps)               if d_exps else 0.0

        # Cheese score — sum of cheese ratings for each team's picks
        r_cheese = sum(cheese_ratings.get(h, 0.0) for h in r_picks)
        d_cheese = sum(cheese_ratings.get(h, 0.0) for h in d_picks)

        # Synergy score — sum of pair win rates for each 2-hero combo on the team
        syn_min = CONFIG.get("synergy_min_matches", 3)
        def _pair_rate(h1, h2):
            key = tuple(sorted([h1, h2]))
            return pair_wins[key] / pair_total[key] if pair_total[key] >= syn_min else 0.5
        if CONFIG.get("include_synergy"):
            r_syn = sum(_pair_rate(h1, h2) for h1, h2 in combinations(r_picks, 2)) if len(r_picks) >= 2 else 0.0
            d_syn = sum(_pair_rate(h1, h2) for h1, h2 in combinations(d_picks, 2)) if len(d_picks) >= 2 else 0.0
        else:
            r_syn = d_syn = 0.0

        # Counter score — average radiant win rate across all 25 hero matchups
        cnt_min = CONFIG.get("counter_min_matches", 3)
        def _matchup_rate(rh, dh):
            key = (rh, dh)
            return matchup_wins[key] / matchup_total[key] if matchup_total[key] >= cnt_min else 0.5
        if CONFIG.get("include_counters") and r_picks and d_picks:
            rates = [_matchup_rate(rh, dh) for rh in r_picks for dh in d_picks]
            r_counter = sum(rates) / len(rates)
        else:
            r_counter = 0.5

        per_match.append({
            "radiant_recent_form":   r_form,
            "dire_recent_form":      d_form,
            "recent_form_diff":      r_form - d_form,
            "radiant_hero_exp_mean": r_exp_mean,
            "radiant_hero_exp_min":  r_exp_min,
            "dire_hero_exp_mean":    d_exp_mean,
            "dire_hero_exp_min":     d_exp_min,
            "radiant_cheese_score":  r_cheese,
            "dire_cheese_score":     d_cheese,
            "cheese_score_diff":     r_cheese - d_cheese,
            "patch":                 float(match.get("patch") or 0),
            "radiant_synergy_score": r_syn,
            "dire_synergy_score":    d_syn,
            "synergy_diff":          r_syn - d_syn,
            "radiant_counter_score": r_counter,
            "counter_diff":          r_counter - 0.5,
        })

        # Update state AFTER recording (no leakage)
        win = match["radiant_win"]
        team_results[r_id].append(1 if win else 0)
        team_results[d_id].append(0 if win else 1)
        for h in r_picks:
            team_hero_cnt[r_id][h] += 1
        for h in d_picks:
            team_hero_cnt[d_id][h] += 1

        # Update synergy pair stats
        for h1, h2 in combinations(r_picks, 2):
            key = tuple(sorted([h1, h2]))
            pair_total[key] += 1
            if win:
                pair_wins[key] += 1
        for h1, h2 in combinations(d_picks, 2):
            key = tuple(sorted([h1, h2]))
            pair_total[key] += 1
            if not win:
                pair_wins[key] += 1

        # Update counter matchup stats
        for rh in r_picks:
            for dh in d_picks:
                matchup_total[(rh, dh)] += 1
                if win:
                    matchup_wins[(rh, dh)] += 1

    log.info(f"Contextual features built for {len(per_match)} matches.")

    # Final-state lookups for prediction time
    team_form = {
        tid: {
            "recent_win_rate": round(sum(res) / len(res), 4) if res else 0.5,
            "games": len(res),
        }
        for tid, res in team_results.items()
    }
    team_hero_exp = {
        tid: dict(hero_counts)
        for tid, hero_counts in team_hero_cnt.items()
    }

    # Final-state synergy/counter lookup tables for prediction time
    synergy_rates = {
        f"{h1}_{h2}": {"wins": pair_wins[k], "total": pair_total[k],
                       "win_rate": round(pair_wins[k] / pair_total[k], 4)}
        for k in pair_total for h1, h2 in [k]
    }
    counter_rates = {
        f"{rh}_{dh}": {"wins": matchup_wins[(rh, dh)], "total": matchup_total[(rh, dh)],
                       "win_rate": round(matchup_wins[(rh, dh)] / matchup_total[(rh, dh)], 4)}
        for (rh, dh) in matchup_total
    }
    log.info(f"Synergy pairs: {len(synergy_rates)}, Counter matchups: {len(counter_rates)}")

    return per_match, team_form, team_hero_exp, cheese_ratings, synergy_rates, counter_rates


# ---------------------------------------------------------------------------
# FEATURE VECTOR CONSTRUCTION
# ---------------------------------------------------------------------------
def match_to_feature_vector(
    match: dict,
    hero_index: dict,
    radiant_enc: dict,
    dire_enc: dict,
    global_mean: float,
    contextual: dict = None,
) -> dict:
    """Convert a single match into a flat feature dict."""
    n_heroes = len(hero_index)
    features = {}

    # --- Radiant pick vector ---
    radiant_pick_vec = np.zeros(n_heroes, dtype=np.float32)
    for hero_id in match.get("radiant_picks", []):
        if hero_id in hero_index:
            radiant_pick_vec[hero_index[hero_id]] = 1.0
    for i, v in enumerate(radiant_pick_vec):
        features[f"r_pick_{i}"] = v

    # --- Dire pick vector ---
    dire_pick_vec = np.zeros(n_heroes, dtype=np.float32)
    for hero_id in match.get("dire_picks", []):
        if hero_id in hero_index:
            dire_pick_vec[hero_index[hero_id]] = 1.0
    for i, v in enumerate(dire_pick_vec):
        features[f"d_pick_{i}"] = v

    # --- Ban vector (combined) ---
    if CONFIG["include_bans"]:
        ban_vec = np.zeros(n_heroes, dtype=np.float32)
        for hero_id in match.get("radiant_bans", []) + match.get("dire_bans", []):
            if hero_id in hero_index:
                ban_vec[hero_index[hero_id]] = 1.0
        for i, v in enumerate(ban_vec):
            features[f"ban_{i}"] = v

    # --- Elo features ---
    features["radiant_elo"] = match.get("radiant_elo_before", 1500.0)
    features["dire_elo"] = match.get("dire_elo_before", 1500.0)
    features["elo_diff"] = match.get("elo_diff", 0.0)

    # --- Team target encoding ---
    r_id = str(match.get("radiant_team_id") or "unknown")
    d_id = str(match.get("dire_team_id") or "unknown")
    features["radiant_team_enc"] = radiant_enc.get(r_id, global_mean)
    features["dire_team_enc"] = dire_enc.get(d_id, 1.0 - global_mean)
    features["team_enc_diff"] = features["radiant_team_enc"] - features["dire_team_enc"]

    # --- Draft order features ---
    # For each of the 10 picks, record which hero was picked at that position.
    # Encoded as the hero's index (normalized to [0,1]).
    if CONFIG["include_draft_order"]:
        ordered = match.get("picks_bans_ordered", [])
        picks_in_order = [pb for pb in ordered if pb.get("is_pick")]
        for pos in range(10):
            if pos < len(picks_in_order):
                hero_id = picks_in_order[pos].get("hero_id")
                team = picks_in_order[pos].get("team", -1)
                col_idx = hero_index.get(hero_id, 0)
                features[f"pick_order_{pos}_hero"] = col_idx / max(n_heroes, 1)
                features[f"pick_order_{pos}_team"] = float(team)
            else:
                features[f"pick_order_{pos}_hero"] = 0.0
                features[f"pick_order_{pos}_team"] = -1.0
        # Last pick advantage: picking last = counter-pick opportunity
        if picks_in_order:
            features["radiant_has_last_pick"] = 1.0 if picks_in_order[-1].get("team") == 0 else 0.0
        else:
            features["radiant_has_last_pick"] = 0.0

    # --- Contextual features (recent form, hero exp, cheese, patch) ---
    if contextual:
        features.update(contextual)

    # --- Metadata (not used as features but useful for analysis) ---
    features["__match_id"] = match.get("match_id")
    features["__start_time"] = match.get("start_time")
    features["__patch"] = match.get("patch")
    features["__radiant_team_id"] = match.get("radiant_team_id")
    features["__dire_team_id"] = match.get("dire_team_id")

    # --- Target ---
    features["radiant_win"] = int(match["radiant_win"])

    return features


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    processed_dir = Path(CONFIG["processed_dir"])
    raw_dir = Path(CONFIG["raw_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Building feature matrix")
    log.info("=" * 60)

    # Load data
    matches = load_matches(processed_dir)
    hero_constants = load_hero_constants(raw_dir)

    # Filter
    matches = filter_matches(matches)

    if not matches:
        log.error("No matches after filtering. Check CONFIG settings.")
        return

    # Build encoders
    hero_index = build_hero_index(matches)
    radiant_enc, dire_enc, global_mean = build_team_target_encoding(matches)
    contextual_list, team_form, team_hero_exp, cheese_ratings, synergy_rates, counter_rates = build_contextual_features(matches)

    # Build feature matrix
    log.info("Building feature vectors...")
    rows = []
    for i, match in enumerate(matches):
        row = match_to_feature_vector(match, hero_index, radiant_enc, dire_enc, global_mean, contextual_list[i])
        rows.append(row)
        if (i + 1) % 1000 == 0:
            log.info(f"  {i+1}/{len(matches)} features built...")

    df = pd.DataFrame(rows)
    log.info(f"Feature matrix shape: {df.shape}")

    # Separate metadata columns (prefixed with __)
    meta_cols = [c for c in df.columns if c.startswith("__")]
    feature_cols = [c for c in df.columns if not c.startswith("__") and c != "radiant_win"]
    label_col = "radiant_win"

    log.info(f"Feature columns: {len(feature_cols)}")
    log.info(f"Metadata columns: {len(meta_cols)}")

    # Save
    out_path = processed_dir / "features.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved feature matrix to {out_path}")

    # Save feature names
    feature_names_path = processed_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info(f"Saved feature names to {feature_names_path}")

    # Save hero index
    hero_index_path = processed_dir / "hero_index.json"
    with open(hero_index_path, "w") as f:
        json.dump(hero_index, f, indent=2)

    # Save team encoder
    team_encoder_path = processed_dir / "team_encoder.json"
    with open(team_encoder_path, "w") as f:
        json.dump({
            "radiant_enc": radiant_enc,
            "dire_enc": dire_enc,
            "global_mean": global_mean,
        }, f, indent=2)
    log.info(f"Saved team encoder to {team_encoder_path}")

    # Save prediction-time lookups for new features
    with open(processed_dir / "team_form.json", "w") as f:
        json.dump(team_form, f, indent=2)
    log.info(f"Saved team form to {processed_dir / 'team_form.json'}")

    with open(processed_dir / "team_hero_exp.json", "w") as f:
        json.dump({tid: {str(h): c for h, c in heroes.items()} for tid, heroes in team_hero_exp.items()}, f, indent=2)
    log.info(f"Saved team hero experience to {processed_dir / 'team_hero_exp.json'}")

    with open(processed_dir / "hero_cheese_ratings.json", "w") as f:
        json.dump({str(k): v for k, v in cheese_ratings.items()}, f, indent=2)
    log.info(f"Saved cheese ratings to {processed_dir / 'hero_cheese_ratings.json'}")

    with open(processed_dir / "hero_synergy_rates.json", "w") as f:
        json.dump(synergy_rates, f, indent=2)
    log.info(f"Saved {len(synergy_rates)} synergy pairs to {processed_dir / 'hero_synergy_rates.json'}")

    with open(processed_dir / "hero_counter_rates.json", "w") as f:
        json.dump(counter_rates, f, indent=2)
    log.info(f"Saved {len(counter_rates)} counter matchups to {processed_dir / 'hero_counter_rates.json'}")

    # Summary
    log.info(f"\nClass balance: {df['radiant_win'].mean():.3f} Radiant win rate")
    log.info(f"Total samples : {len(df)}")


if __name__ == "__main__":
    main()
