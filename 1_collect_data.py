"""
1_collect_data.py
=================
Scrapes pro match data and team information from the OpenDota API.
Prioritises Tier 1 tournament matches (Majors, TI, DPC Division 1).

What this script does:
  1. Fetches all leagues from /leagues and filters to premium/professional tiers
  2. For each league, fetches its match list via /leagues/{id}/matches
  3. For each match, fetches full details via /matches/{id}
     - Extracts: picks/bans, radiant_win, team IDs, patch, start_time, league
  4. Fetches all team metadata (via /teams)
  5. Saves everything to disk under data/raw/

League tiers (OpenDota):
  "premium"      — The International, Majors (highest quality)
  "professional" — DPC Division 1, top regional leagues
  "amateur"      — Lower division, qualifiers (excluded by default)

Usage:
  python 1_collect_data.py

Rate limits (free tier):
  60 requests/minute, ~50,000 requests/month
  This script respects those limits automatically.
"""

import json
import time
import logging
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    # Which league tiers to include. Options: "premium", "professional", "amateur"
    # "premium"      = TI + Majors only (~highest quality, fewer matches)
    # "professional" = adds DPC Division 1 and top regional leagues (recommended)
    "league_tiers": ["premium", "professional"],

    # Hard cap on total matches downloaded (to stay within API limits).
    # At 1.05s/request, 10,000 matches takes ~3 hours.
    # Set to None for no limit (will collect everything in the tier).
    "target_match_count": 10_000,

    # Only download matches on or after this date (YYYY-MM-DD).
    # Saves API calls by skipping old matches entirely.
    # Set to None to download all matches.
    "min_date": "2023-01-01",

    # Minimum match ID to fetch. Match IDs are sequential so this is a fast
    # pre-filter that requires no API call — IDs below this are skipped instantly.
    # Useful if you know the approximate ID boundary for your desired date range.
    # Set to None to disable.
    "min_match_id": 8142981335,

    # Minimum league ID to scan. League IDs are sequential (higher = newer).
    # Leagues below this ID are skipped before any API call is made.
    # ~14000 corresponds to early 2023 (TI12 era). Set to None to disable.
    "min_league_id": 16000,

    # OpenDota API key (optional but raises rate limits significantly).
    # Get one free at https://www.opendota.com/api-keys
    # Leave as None to use the free unauthenticated tier.
    "api_key": None,

    # Seconds to wait between requests.
    # 1.05s keeps you safely under 60 req/min on free tier.
    # Reduce to 0.5 if you have an API key.
    "request_delay": 1.05,

    # Where to save raw data.
    "raw_dir": "D:/Dota 2 python data/raw",

    # Resume from last run — skips match IDs already saved to disk.
    "resume": True,
}

BASE_URL = "https://api.opendota.com/api"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
Path("data").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/collect_data.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def api_get(endpoint: str, params: dict = None, retries: int = 5) -> dict | list | None:
    """GET from OpenDota API with retry/backoff logic."""
    if params is None:
        params = {}
    if CONFIG["api_key"]:
        params["api_key"] = CONFIG["api_key"]

    url = f"{BASE_URL}{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
            elif resp.status_code >= 500:
                wait = 10 * (attempt + 1)
                log.warning(f"Server error {resp.status_code}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"Unexpected status {resp.status_code} for {url}")
                return None
        except requests.exceptions.RequestException as e:
            wait = 10 * (attempt + 1)
            log.warning(f"Request error: {e}. Waiting {wait}s...")
            time.sleep(wait)
    log.error(f"Failed after {retries} retries: {url}")
    return None


def load_existing_match_ids(raw_dir: Path) -> set:
    """Return set of match IDs already saved to disk."""
    matches_dir = raw_dir / "matches"
    if not matches_dir.exists():
        return set()
    return {int(f.stem) for f in matches_dir.glob("*.json")}


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# STEP 1a: Fetch and filter leagues by tier
# ---------------------------------------------------------------------------
def collect_leagues(raw_dir: Path) -> list[dict]:
    """
    Fetch all leagues and filter to the configured tiers.
    Saves the full league list to disk and returns filtered leagues.
    """
    log.info("Fetching league list from /leagues...")
    leagues = api_get("/leagues")
    time.sleep(CONFIG["request_delay"])

    if not leagues:
        log.error("Failed to fetch leagues.")
        return []

    # Save full league list for reference
    save_json(raw_dir / "leagues.json", leagues)
    log.info(f"Fetched {len(leagues)} total leagues.")

    # Filter to configured tiers
    target_tiers = set(CONFIG["league_tiers"])
    filtered = [l for l in leagues if l.get("tier") in target_tiers]

    # Log breakdown by tier
    tier_counts = {}
    for l in filtered:
        t = l.get("tier", "unknown")
        tier_counts[t] = tier_counts.get(t, 0) + 1
    for tier, count in sorted(tier_counts.items()):
        log.info(f"  {tier}: {count} leagues")

    log.info(f"Filtered to {len(filtered)} leagues across tiers: {target_tiers}")
    return filtered


# ---------------------------------------------------------------------------
# STEP 1b: Collect match IDs from filtered leagues
# ---------------------------------------------------------------------------
def collect_match_ids_from_leagues(leagues: list[dict], existing_ids: set) -> list[int]:
    """
    For each league, fetch the list of match IDs via /leagues/{id}/matches.
    Returns all new match IDs not already downloaded, sorted newest first.
    """
    import datetime
    min_timestamp = None
    if CONFIG.get("min_date"):
        min_timestamp = datetime.datetime.strptime(CONFIG["min_date"], "%Y-%m-%d").replace(
            tzinfo=datetime.timezone.utc
        ).timestamp()

    all_match_ids = []
    seen = set(existing_ids)

    min_league_id = CONFIG.get("min_league_id")
    if min_league_id:
        before = len(leagues)
        leagues = [l for l in leagues if (l.get("league_id") or l.get("leagueid") or 0) >= min_league_id]
        log.info(f"Min league ID filter ({min_league_id}): skipping {before - len(leagues)} old leagues, {len(leagues)} remain.")

    total_leagues = len(leagues)
    for i, league in enumerate(leagues):
        league_id = league.get("league_id") or league.get("leagueid")
        league_name = league.get("name", f"League {league_id}")
        tier = league.get("tier", "unknown")

        if not league_id:
            continue

        log.info(f"  [{i+1}/{total_leagues}] Fetching matches for: {league_name} (tier={tier}, id={league_id})")

        matches = api_get(f"/leagues/{league_id}/matches")
        time.sleep(CONFIG["request_delay"])

        if not matches:
            log.warning(f"    No matches returned for league {league_id}.")
            continue

        # Filter by min_date if start_time is available in the response
        if min_timestamp:
            before = len(matches)
            matches = [m for m in matches if (m.get("start_time") or 0) >= min_timestamp]
            if len(matches) < before:
                log.info(f"    Date filter removed {before - len(matches)} matches before {CONFIG['min_date']}.")

        min_id = CONFIG.get("min_match_id")
        new_ids = [m["match_id"] for m in matches if m["match_id"] not in seen and (not min_id or m["match_id"] >= min_id)]
        for mid in new_ids:
            seen.add(mid)
        all_match_ids.extend(new_ids)

        log.info(f"    Found {len(matches)} matches, {len(new_ids)} new. Running total: {len(all_match_ids)}")

    # Sort newest first (higher match_id = more recent) before applying cap,
    # so the cap keeps the most recent N matches across all leagues.
    all_match_ids.sort(reverse=True)

    target = CONFIG["target_match_count"]
    if target:
        all_match_ids = all_match_ids[:target]

    log.info(f"Total match IDs to download: {len(all_match_ids)}")
    return all_match_ids


# ---------------------------------------------------------------------------
# STEP 2: Fetch and save full match details
# ---------------------------------------------------------------------------
def extract_match_fields(raw: dict) -> dict | None:
    """
    Extract only the fields we need from a raw match API response.
    Returns None if the match is missing essential data.
    """
    # Essential fields
    match_id = raw.get("match_id")
    radiant_win = raw.get("radiant_win")
    picks_bans = raw.get("picks_bans")
    start_time = raw.get("start_time")
    patch = raw.get("patch")

    # Team IDs — may be None for unnamed stacks
    radiant_team_id = raw.get("radiant_team_id") or raw.get("radiant_team", {}).get("team_id")
    dire_team_id = raw.get("dire_team_id") or raw.get("dire_team", {}).get("team_id")
    radiant_team_name = raw.get("radiant_name") or (raw.get("radiant_team") or {}).get("name", "Unknown")
    dire_team_name = raw.get("dire_name") or (raw.get("dire_team") or {}).get("name", "Unknown")

    # Validate essential fields
    if radiant_win is None:
        return None  # Match may be incomplete/abandoned
    if not picks_bans:
        return None  # No draft data — can't use this match

    # Separate picks from bans, and by team
    picks = [pb for pb in picks_bans if not pb.get("is_pick") is False and pb.get("is_pick")]
    bans = [pb for pb in picks_bans if not pb.get("is_pick")]

    radiant_picks = [pb["hero_id"] for pb in picks if pb.get("team") == 0]
    dire_picks = [pb["hero_id"] for pb in picks if pb.get("team") == 1]
    radiant_bans = [pb["hero_id"] for pb in bans if pb.get("team") == 0]
    dire_bans = [pb["hero_id"] for pb in bans if pb.get("team") == 1]

    # Need at least 5 picks per side to be a valid match
    if len(radiant_picks) < 5 or len(dire_picks) < 5:
        return None

    return {
        "match_id": match_id,
        "start_time": start_time,
        "patch": patch,
        "radiant_win": radiant_win,
        "radiant_team_id": radiant_team_id,
        "dire_team_id": dire_team_id,
        "radiant_team_name": radiant_team_name,
        "dire_team_name": dire_team_name,
        "radiant_picks": radiant_picks,
        "dire_picks": dire_picks,
        "radiant_bans": radiant_bans,
        "dire_bans": dire_bans,
        "picks_bans_ordered": picks_bans,  # Keep full ordered draft for order-aware features
        "league_id": raw.get("leagueid"),
    }


def download_match_details(match_ids: list[int], raw_dir: Path) -> None:
    """Fetch and save full match details for each match ID."""
    import datetime
    matches_dir = raw_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    min_timestamp = None
    if CONFIG.get("min_date"):
        min_timestamp = datetime.datetime.strptime(CONFIG["min_date"], "%Y-%m-%d").replace(
            tzinfo=datetime.timezone.utc
        ).timestamp()
        log.info(f"Date filter active: skipping matches before {CONFIG['min_date']}")

    min_id = CONFIG.get("min_match_id")
    if min_id:
        before = len(match_ids)
        match_ids = [mid for mid in match_ids if mid >= min_id]
        log.info(f"Min match ID filter ({min_id}): {before - len(match_ids)} IDs removed, {len(match_ids)} remain.")

    total = len(match_ids)
    saved = 0
    skipped = 0
    date_skipped = 0

    for i, match_id in enumerate(match_ids):
        save_path = matches_dir / f"{match_id}.json"

        if save_path.exists() and CONFIG["resume"]:
            skipped += 1
            continue

        raw = api_get(f"/matches/{match_id}")
        time.sleep(CONFIG["request_delay"])

        if raw is None:
            log.warning(f"  [{i+1}/{total}] Failed to fetch match {match_id}. Skipping.")
            continue

        if min_timestamp and (raw.get("start_time") or 0) < min_timestamp:
            date_skipped += 1
            save_json(save_path, {"match_id": match_id, "invalid": True})
            continue

        extracted = extract_match_fields(raw)
        if extracted is None:
            log.info(f"  [{i+1}/{total}] Match {match_id} missing required fields. Skipping.")
            # Save a marker so we don't retry
            save_json(save_path, {"match_id": match_id, "invalid": True})
            continue

        save_json(save_path, extracted)
        saved += 1

        if (i + 1) % 100 == 0:
            log.info(f"  Progress: {i+1}/{total} processed, {saved} saved, {skipped} skipped.")

    log.info(f"Match download complete. Saved: {saved}, Skipped (existing): {skipped}, Skipped (too old): {date_skipped}")


# ---------------------------------------------------------------------------
# STEP 3: Fetch team metadata
# ---------------------------------------------------------------------------
def collect_team_metadata(raw_dir: Path) -> None:
    """Fetch all teams from /teams and save to disk."""
    log.info("Fetching team metadata from /teams...")
    teams = api_get("/teams")
    if not teams:
        log.error("Failed to fetch teams.")
        return

    save_path = raw_dir / "teams.json"
    save_json(save_path, teams)
    log.info(f"Saved {len(teams)} teams to {save_path}")


# ---------------------------------------------------------------------------
# STEP 4: Fetch hero constants (hero ID → name mapping)
# ---------------------------------------------------------------------------
def collect_hero_constants(raw_dir: Path) -> None:
    """Fetch hero ID → name mapping from OpenDota constants."""
    log.info("Fetching hero constants...")
    heroes = api_get("/heroes")
    if not heroes:
        log.error("Failed to fetch hero constants.")
        return

    # Build a clean id → name map
    hero_map = {h["id"]: h["localized_name"] for h in heroes}
    save_path = raw_dir / "hero_constants.json"
    save_json(save_path, hero_map)
    log.info(f"Saved {len(hero_map)} heroes to {save_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    raw_dir = Path(CONFIG["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Starting Dota 2 pro match data collection")
    log.info(f"  League tiers   : {CONFIG['league_tiers']}")
    log.info(f"  Target matches : {CONFIG['target_match_count']}")
    log.info(f"  API key        : {'set' if CONFIG['api_key'] else 'not set (free tier)'}")
    log.info(f"  Request delay  : {CONFIG['request_delay']}s")
    log.info(f"  Resume mode    : {CONFIG['resume']}")
    log.info("=" * 60)

    # Fetch hero map and teams first (fast, one-off calls)
    collect_hero_constants(raw_dir)
    collect_team_metadata(raw_dir)

    # Load already-downloaded match IDs to support resuming
    existing_ids = load_existing_match_ids(raw_dir) if CONFIG["resume"] else set()
    log.info(f"Found {len(existing_ids)} already-downloaded matches on disk.")

    # Check for a cached match ID queue from a previous interrupted run
    queue_cache_path = raw_dir / "match_id_queue.json"
    if CONFIG["resume"] and queue_cache_path.exists():
        cached = load_json(queue_cache_path)
        # Remove any IDs that have since been downloaded
        match_ids = [mid for mid in cached if mid not in existing_ids]
        log.info(f"Loaded {len(cached)} match IDs from queue cache, {len(match_ids)} still pending.")
    else:
        # Step 1: Get leagues filtered by tier
        leagues = collect_leagues(raw_dir)
        if not leagues:
            log.error("No leagues found for the configured tiers. Check CONFIG['league_tiers'].")
            return

        # Step 2: Collect match IDs from those leagues
        match_ids = collect_match_ids_from_leagues(leagues, existing_ids)

        # Save the queue so interrupted runs can resume without re-scanning leagues
        if match_ids:
            save_json(queue_cache_path, match_ids)
            log.info(f"Saved match ID queue ({len(match_ids)} IDs) to {queue_cache_path}")

    if not match_ids:
        log.info("No new matches to download.")
        return

    # Step 3: Download match details
    download_match_details(match_ids, raw_dir)

    # All done — remove the queue cache so the next run does a fresh league scan
    if queue_cache_path.exists():
        queue_cache_path.unlink()
        log.info("Queue cache cleared. Next run will re-scan leagues for new matches.")

    log.info("Data collection complete.")
    log.info(f"Raw data saved to: {raw_dir.resolve()}")


if __name__ == "__main__":
    main()
