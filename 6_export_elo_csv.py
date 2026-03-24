"""
6_export_elo_csv.py
===================
Reads the Elo ratings built by 2_build_elo.py and exports a simple CSV
sorted by Elo (highest first).

Output:
  data/elo_rankings.csv
"""

import csv
import json
import logging
from pathlib import Path

CONFIG = {
    "elo_ratings_path": "data/processed/elo_ratings.json",
    "output_path": "data/elo_rankings.csv",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    elo_path = Path(CONFIG["elo_ratings_path"])
    if not elo_path.exists():
        raise FileNotFoundError(f"Elo ratings not found: {elo_path}. Run 2_build_elo.py first.")

    with open(elo_path) as f:
        elo_data = json.load(f)

    rows = [
        {
            "team_name": info["team_name"],
            "team_id": tid,
            "elo": round(info["elo"], 2),
            "matches_played": info["matches_played"],
        }
        for tid, info in elo_data.items()
    ]

    rows.sort(key=lambda r: r["elo"], reverse=True)

    out_path = Path(CONFIG["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["team_name", "team_id", "elo", "matches_played"])
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Exported {len(rows)} teams to {out_path}")


if __name__ == "__main__":
    main()
