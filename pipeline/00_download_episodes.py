from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import feedparser
import requests
import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[-\s]+", "_", value)
    return value[:120]


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_file(url: str, destination: Path, timeout_seconds: int = 60) -> None:
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main() -> None:
    config = load_config()
    rss_url = config["podcast"]["rss_url"]
    audio_dir = ensure_dir(config["paths"]["audio_dir"])
    limit = int(config["download"].get("limit", 5))
    overwrite = bool(config["download"].get("overwrite", False))
    timeout_seconds = int(config["download"].get("timeout_seconds", 60))

    print(f"Načítám RSS feed: {rss_url}")
    feed = feedparser.parse(rss_url)

    if getattr(feed, "bozo", False):
        print("Pozor: feedparser hlásí problém při parsování feedu.")
        if hasattr(feed, "bozo_exception"):
            print(feed.bozo_exception)

    entries = feed.entries[:limit]
    print(f"Nalezeno epizod ke stažení: {len(entries)}")

    for i, entry in enumerate(entries, start=1):
        title = entry.get("title", f"episode_{i}")
        enclosure_url = None

        enclosures = entry.get("enclosures", [])
        if enclosures:
            enclosure_url = enclosures[0].get("href") or enclosures[0].get("url")

        if not enclosure_url:
            print(f"❌ Přeskakuji '{title}' - chybí MP3 URL")
            continue

        file_name = f"{i:03d}_{slugify(title)}.mp3"
        destination = audio_dir / file_name

        if destination.exists() and not overwrite:
            print(f"⏭️ Už existuje, přeskakuji: {destination.name}")
            continue

        print(f"⬇️ Stahuji: {title}")
        print(f"   URL: {enclosure_url}")

        try:
            download_file(enclosure_url, destination, timeout_seconds=timeout_seconds)
            print(f"✅ Uloženo do: {destination}")
        except Exception as e:
            print(f"❌ Chyba při stahování '{title}': {e}")


if __name__ == "__main__":
    main()