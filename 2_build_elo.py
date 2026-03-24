"""
2_build_elo.py
==============
Builds a chronological Elo rating for every pro team using the match data
collected by 1_collect_data.py.

Key design decisions:
  - Matches are processed in chronological order (oldest first) so that each
    team's Elo at the time of a match reflects only their *prior* results.
  - The Elo BEFORE each match is stored alongside that match — this is what
    gets used as a model feature (no data leakage).
  - Teams with no ID (unnamed stacks) are assigned a static default Elo of 1500
    and are excluded from Elo updates.
  - K-factor of 32 is standard for competitive gaming.

Outputs:
  data/processed/elo_ratings.json   — final Elo rating per team_id
  data/processed/matches_with_elo.json — match records with pre-match Elo attached
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "raw_dir": "D:/Dota 2 python data/raw",
    "processed_dir": "data/processed",

    # Starting Elo for all teams
    "initial_elo": 1500.0,

    # K-factor: how much each match shifts the Elo rating.
    # 32 is standard. Increase for faster adaptation, decrease for stability.
    "k_factor": 32,

    # Minimum matches a team must have played before their Elo is considered
    # 'reliable'. Teams below this threshold are flagged in the output.
    "min_matches_reliable": 10,
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
        logging.FileHandler("data/build_elo.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ELO FUNCTIONS
# ---------------------------------------------------------------------------
def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given ratings of A and B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: float) -> tuple[float, float]:
    """
    Update Elo ratings for two players after a match.
    score_a: 1.0 if A won, 0.0 if A lost.
    Returns (new_rating_a, new_rating_b).
    """
    e_a = expected_score(rating_a, rating_b)
    e_b = 1.0 - e_a
    score_b = 1.0 - score_a

    new_a = rating_a + k * (score_a - e_a)
    new_b = rating_b + k * (score_b - e_b)
    return new_a, new_b


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_all_matches(raw_dir: Path) -> list[dict]:
    """Load all valid match JSON files from disk."""
    matches_dir = raw_dir / "matches"
    matches = []
    for f in matches_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
        # Skip invalid/incomplete matches
        if data.get("invalid"):
            continue
        if data.get("radiant_win") is None:
            continue
        if not data.get("radiant_picks") or not data.get("dire_picks"):
            continue
        matches.append(data)

    log.info(f"Loaded {len(matches)} valid matches from disk.")
    return matches


def load_teams(raw_dir: Path) -> dict:
    """Load team metadata, keyed by team_id."""
    path = raw_dir / "teams.json"
    if not path.exists():
        log.warning("teams.json not found. Team name lookup will be unavailable.")
        return {}
    with open(path) as f:
        teams = json.load(f)
    return {str(t["team_id"]): t for t in teams}


# ---------------------------------------------------------------------------
# MAIN ELO BUILD
# ---------------------------------------------------------------------------
def build_elo_ratings(matches: list[dict]) -> list[dict]:
    """
    Process all matches chronologically and compute pre-match Elo for each team.

    Returns:
        List of match dicts enriched with:
          - radiant_elo_before  : Radiant's Elo immediately before this match
          - dire_elo_before     : Dire's Elo immediately before this match
          - elo_diff            : radiant_elo_before - dire_elo_before
          - radiant_elo_after   : Radiant's Elo after this match (for tracking)
          - dire_elo_after      : Dire's Elo after this match (for tracking)
    """
    # Sort by start_time ascending (chronological order)
    matches_sorted = sorted(matches, key=lambda m: m.get("start_time") or 0)
    log.info(f"Processing {len(matches_sorted)} matches in chronological order...")

    # Current Elo for each team_id (string key)
    elo: dict[str, float] = defaultdict(lambda: CONFIG["initial_elo"])

    # Match counts per team (for reliability flagging)
    match_count: dict[str, int] = defaultdict(int)

    enriched_matches = []

    for i, match in enumerate(matches_sorted):
        radiant_id = str(match.get("radiant_team_id") or "unknown_radiant")
        dire_id = str(match.get("dire_team_id") or "unknown_dire")
        radiant_win = match["radiant_win"]

        # Record Elo BEFORE this match (this is what becomes the model feature)
        r_elo_before = elo[radiant_id]
        d_elo_before = elo[dire_id]

        # Update Elo for named teams only
        radiant_has_id = match.get("radiant_team_id") is not None
        dire_has_id = match.get("dire_team_id") is not None

        if radiant_has_id and dire_has_id:
            score_radiant = 1.0 if radiant_win else 0.0
            new_r, new_d = update_elo(r_elo_before, d_elo_before, score_radiant, CONFIG["k_factor"])
            elo[radiant_id] = new_r
            elo[dire_id] = new_d
            match_count[radiant_id] += 1
            match_count[dire_id] += 1
        else:
            new_r, new_d = r_elo_before, d_elo_before  # No update for unknown teams

        # Enrich the match record
        enriched = {
            **match,
            "radiant_elo_before": round(r_elo_before, 2),
            "dire_elo_before": round(d_elo_before, 2),
            "elo_diff": round(r_elo_before - d_elo_before, 2),
            "radiant_elo_after": round(new_r, 2),
            "dire_elo_after": round(new_d, 2),
            "radiant_elo_reliable": match_count[radiant_id] >= CONFIG["min_matches_reliable"],
            "dire_elo_reliable": match_count[dire_id] >= CONFIG["min_matches_reliable"],
        }
        enriched_matches.append(enriched)

        if (i + 1) % 1000 == 0:
            log.info(f"  Processed {i+1}/{len(matches_sorted)} matches...")

    log.info(f"Elo computation complete. {len(enriched_matches)} matches enriched.")
    return enriched_matches, dict(elo), dict(match_count)


def log_top_teams(elo: dict, match_count: dict, teams_meta: dict, top_n: int = 20):
    """Log the top N teams by Elo rating."""
    # Filter to teams with enough matches to be reliable
    reliable = {
        tid: rating for tid, rating in elo.items()
        if match_count.get(tid, 0) >= CONFIG["min_matches_reliable"]
    }
    sorted_teams = sorted(reliable.items(), key=lambda x: x[1], reverse=True)

    log.info(f"\nTop {top_n} teams by Elo rating (min {CONFIG['min_matches_reliable']} matches):")
    log.info(f"{'Rank':<5} {'Team Name':<35} {'Team ID':<12} {'Elo':>8} {'Matches':>8}")
    log.info("-" * 72)
    for rank, (tid, rating) in enumerate(sorted_teams[:top_n], 1):
        team_meta = teams_meta.get(tid, {})
        name = team_meta.get("name", "Unknown")
        matches = match_count.get(tid, 0)
        log.info(f"{rank:<5} {name:<35} {tid:<12} {rating:>8.1f} {matches:>8}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    raw_dir = Path(CONFIG["raw_dir"])
    processed_dir = Path(CONFIG["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Building Elo ratings from pro match data")
    log.info(f"  Initial Elo  : {CONFIG['initial_elo']}")
    log.info(f"  K-factor     : {CONFIG['k_factor']}")
    log.info("=" * 60)

    # Load data
    matches = load_all_matches(raw_dir)
    teams_meta = load_teams(raw_dir)

    if not matches:
        log.error("No matches found. Run 1_collect_data.py first.")
        return

    # Build Elo
    enriched_matches, final_elo, match_count = build_elo_ratings(matches)

    # Save enriched matches
    matches_out_path = processed_dir / "matches_with_elo.json"
    with open(matches_out_path, "w") as f:
        json.dump(enriched_matches, f)
    log.info(f"Saved {len(enriched_matches)} enriched matches to {matches_out_path}")

    # Save final Elo ratings
    elo_out = {
        tid: {
            "elo": round(rating, 2),
            "matches_played": match_count.get(tid, 0),
            "reliable": match_count.get(tid, 0) >= CONFIG["min_matches_reliable"],
            "team_name": teams_meta.get(tid, {}).get("name", "Unknown"),
        }
        for tid, rating in final_elo.items()
    }
    elo_out_path = processed_dir / "elo_ratings.json"
    with open(elo_out_path, "w") as f:
        json.dump(elo_out, f, indent=2)
    log.info(f"Saved final Elo ratings for {len(elo_out)} teams to {elo_out_path}")

    # Print leaderboard
    log_top_teams(final_elo, match_count, teams_meta)

    # Summary stats
    elo_values = list(final_elo.values())
    log.info(f"\nElo summary:")
    log.info(f"  Teams tracked : {len(elo_values)}")
    log.info(f"  Elo range     : {min(elo_values):.1f} – {max(elo_values):.1f}")
    log.info(f"  Mean Elo      : {sum(elo_values)/len(elo_values):.1f}")


if __name__ == "__main__":
    main()
