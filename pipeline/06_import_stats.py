from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_date(date_str: str) -> str:
    """Normalize date to YYYY-MM-DD."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str.strip()


def extract_episode_code(title: str) -> str:
    """Extract episode code like S14E06 from title."""
    match = re.search(r"(S\d+E\d+)", title, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def load_spotify_episodes(path: Path) -> dict[str, dict]:
    episodes = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            title = row["Episode name"].strip()
            episodes[title] = {
                "title": title,
                "episode_code": extract_episode_code(title),
                "release_date": parse_date(row["Publish date"]),
                "spotify_plays": int(row["Plays & downloads"]),
            }
    return episodes


def load_apple_episodes(path: Path) -> dict[str, dict]:
    episodes = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            title = row["Episode Title"].strip()
            plays_str = row.get("Plays", "").strip()
            listeners_str = row.get("Unique Listeners", "").strip()
            engaged_str = row.get("Unique Engaged Listeners", "").strip()
            avg_str = row.get("Average Consumption", "").strip()
            dur_str = row.get("Duration", "").strip()
            episodes[title] = {
                "title": title,
                "episode_code": extract_episode_code(title),
                "release_date": parse_date(row["Release Date"]),
                "apple_plays": int(plays_str) if plays_str not in ["-", "", "NaN"] else 0,
                "apple_listeners": int(listeners_str) if listeners_str not in ["-", "", "NaN"] else 0,
                "apple_engaged_listeners": int(engaged_str) if engaged_str not in ["-", "", "NaN"] else 0,
                "apple_avg_consumption": round(float(avg_str), 3) if avg_str not in ["-", "", "NaN"] else None,
                "apple_duration_seconds": int(float(dur_str)) if dur_str not in ["-", "", "NaN"] else 0,
            }
    return episodes


def merge_episodes(spotify: dict, apple: dict) -> list[dict]:
    """Merge Spotify and Apple data by title."""
    all_titles = set(spotify.keys()) | set(apple.keys())
    merged = []

    for title in all_titles:
        sp = spotify.get(title, {})
        ap = apple.get(title, {})

        episode = {
            "title": title,
            "episode_code": sp.get("episode_code") or ap.get("episode_code", ""),
            "release_date": sp.get("release_date") or ap.get("release_date", ""),
            # Spotify
            "spotify_plays": sp.get("spotify_plays", 0),
            # Apple
            "apple_plays": ap.get("apple_plays", 0),
            "apple_listeners": ap.get("apple_listeners", 0),
            "apple_engaged_listeners": ap.get("apple_engaged_listeners", 0),
            "apple_avg_consumption": ap.get("apple_avg_consumption"),
            "apple_duration_seconds": ap.get("apple_duration_seconds", 0),
            # Combined
            "total_plays": sp.get("spotify_plays", 0) + ap.get("apple_plays", 0),
            "total_audience": ap.get("apple_listeners", 0),
        }
        merged.append(episode)

    # Sort by release date
    merged.sort(key=lambda x: x["release_date"])
    return merged


def load_spotify_performance_daily(path: Path) -> list[dict]:
    """Load daily Spotify performance data (Plays & downloads + Audience)."""
    series = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            plays_str = row.get("Plays & downloads", "").strip()
            audience_str = row.get("Audience", "").strip()
            series.append({
                "date": parse_date(row["Date"]),
                "plays": int(plays_str) if plays_str else 0,
                "audience": int(audience_str) if audience_str else 0,
            })
    series.sort(key=lambda x: x["date"])
    return series


def load_spotify_engagement_daily(path: Path) -> list[dict]:
    """Load daily Spotify engagement data (consumption time, followers)."""
    series = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            consumption_str = row.get("Consumption time (hours)", "").strip()
            followers_str = row.get("Followers", "").strip()
            series.append({
                "date": parse_date(row["Date"]),
                "consumption_hours": round(float(consumption_str), 2) if consumption_str else 0,
                "followers": int(followers_str) if followers_str else 0,
            })
    series.sort(key=lambda x: x["date"])
    return series


def load_apps(path: Path) -> list[dict]:
    """Load app distribution data."""
    apps = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            apps.append({
                "app": row["App"].strip(),
                "percentage": round(float(row["Percentage"]), 2),
            })
    apps.sort(key=lambda x: x["percentage"], reverse=True)
    return apps


def load_geo(path: Path) -> list[dict]:
    """Load geographic distribution."""
    geo = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            geo.append({
                "country": row["Geo"].strip(),
                "percentage": round(float(row["Percentage"]), 4),
            })
    geo.sort(key=lambda x: x["percentage"], reverse=True)
    return geo


def compute_summary(episodes: list[dict], daily: list[dict]) -> dict:
    main = [e for e in episodes if e["episode_code"]]  # exclude bonus/teasers

    total_spotify = sum(e["spotify_plays"] for e in main)
    total_apple = sum(e["apple_plays"] for e in main)
    total = total_spotify + total_apple

    # Season-level aggregation
    seasons: dict[str, dict] = {}
    for ep in main:
        code = ep["episode_code"]
        m = re.match(r"S(\d+)E\d+", code, re.IGNORECASE)
        if not m:
            continue
        season = f"S{int(m.group(1)):02d}"
        if season not in seasons:
            seasons[season] = {"season": season, "episodes": 0, "total_plays": 0, "spotify_plays": 0, "apple_plays": 0}
        seasons[season]["episodes"] += 1
        seasons[season]["total_plays"] += ep["total_plays"]
        seasons[season]["spotify_plays"] += ep["spotify_plays"]
        seasons[season]["apple_plays"] += ep["apple_plays"]

    # Growth: compare first half vs second half of daily plays
    mid = len(daily) // 2
    first_half_avg = sum(d["plays"] for d in daily[:mid]) / max(mid, 1)
    second_half_avg = sum(d["plays"] for d in daily[mid:]) / max(len(daily) - mid, 1)
    growth_pct = round((second_half_avg - first_half_avg) / max(first_half_avg, 1) * 100, 1)

    return {
        "total_plays": total,
        "total_spotify_plays": total_spotify,
        "total_apple_plays": total_apple,
        "spotify_share_pct": round(total_spotify / max(total, 1) * 100, 1),
        "apple_share_pct": round(total_apple / max(total, 1) * 100, 1),
        "total_episodes": len(main),
        "avg_plays_per_episode": round(total / max(len(main), 1), 1),
        "seasons": sorted(seasons.values(), key=lambda x: x["season"]),
        "growth_pct_vs_first_half": growth_pct,
    }


def main() -> None:
    config = load_config()
    stats_dir = Path("data/stats")
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Načítám Spotify data...")
    spotify_eps = load_spotify_episodes(stats_dir / "spotify_episodes.csv")
    print(f"   ✅ {len(spotify_eps)} epizod ze Spotify")

    print("🍎 Načítám Apple Podcasts data...")
    apple_eps = load_apple_episodes(stats_dir / "apple_episodes.csv")
    print(f"   ✅ {len(apple_eps)} epizod z Apple")

    print("📈 Načítám denní výkon (Spotify Performance)...")
    daily = load_spotify_performance_daily(stats_dir / "spotify_streams_daily.csv")
    print(f"   ✅ {len(daily)} dní dat")

    print("💬 Načítám denní engagement...")
    engagement_daily = load_spotify_engagement_daily(stats_dir / "spotify_engagement_daily.csv")
    print(f"   ✅ {len(engagement_daily)} dní engagement dat")

    print("📱 Načítám distribuci aplikací...")
    apps = load_apps(stats_dir / "spotify_apps.csv")
    print(f"   ✅ {len(apps)} aplikací")

    print("🌍 Načítám geografii...")
    geo = load_geo(stats_dir / "spotify_geo.csv")
    print(f"   ✅ {len(geo)} zemí")

    print("\n🔀 Slučuji data...")
    episodes = merge_episodes(spotify_eps, apple_eps)
    summary = compute_summary(episodes, daily)

    output = {
        "summary": summary,
        "episodes": episodes,
        "spotify_daily": daily,
        "spotify_engagement_daily": engagement_daily,
        "apps": apps,
        "geo": geo,
    }

    output_path = processed_dir / "listening_stats.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Shrnutí:")
    print(f"   Celkem přehrání: {summary['total_plays']:,}")
    print(f"   Spotify: {summary['total_spotify_plays']:,} ({summary['spotify_share_pct']}%)")
    print(f"   Apple:   {summary['total_apple_plays']:,} ({summary['apple_share_pct']}%)")
    print(f"   Průměr na epizodu: {summary['avg_plays_per_episode']:,}")
    print(f"   Růst (2. vs 1. polovina): +{summary['growth_pct_vs_first_half']}%")
    print(f"\n🎉 listening_stats.json uložen: {output_path}")


if __name__ == "__main__":
    main()
