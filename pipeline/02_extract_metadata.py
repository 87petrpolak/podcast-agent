from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import feedparser
import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Czech IT topic keywords — used to tag episodes
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "AI & ML": [
        "umělá inteligence", "strojové učení", "machine learning", "deep learning",
        "neural", "llm", "gpt", "claude", "openai", "anthropic", "model", "ai agent",
        "embedding", "rag", "prompt", "token", "inference", "fine-tuning",
    ],
    "Startupy": [
        "startup", "fundraising", "investor", "seed", "série a", "série b",
        "vc", "venture capital", "exit", "akvizice", "pivot", "product-market fit",
        "zakladatel", "founder", "co-founder",
    ],
    "Cloud & DevOps": [
        "cloud", "aws", "azure", "gcp", "google cloud", "kubernetes", "docker",
        "devops", "ci/cd", "terraform", "infrastructure", "microservices", "serverless",
        "deployment", "pipeline", "monitoring",
    ],
    "Architektura": [
        "architektura", "architecture", "event-driven", "event sourcing", "cqrs",
        "domain-driven", "ddd", "microservices", "monolith", "api", "rest", "graphql",
        "adr", "architecture decision", "governance",
    ],
    "Kariéra": [
        "kariéra", "career", "práce", "job", "přechod", "senior", "junior", "cto",
        "leadership", "management", "tým", "team", "hire", "nábor", "remote", "hybrid",
    ],
    "Bezpečnost": [
        "bezpečnost", "security", "kybernetická", "cyber", "hack", "penetrační",
        "vulnerability", "zero-day", "gdpr", "compliance", "šifrování", "encryption",
    ],
    "Data & Analytics": [
        "data", "analytics", "datová", "warehouse", "lakehouse", "etl", "pipeline",
        "bi", "business intelligence", "dashboard", "sql", "databáze", "database",
        "keboola", "snowflake", "databricks",
    ],
    "Podnikání & Produkt": [
        "produkt", "product", "produktový", "saas", "b2b", "b2c", "zákazník",
        "customer", "ux", "design", "roadmap", "sprint", "agile", "scrum",
    ],
    "Česká IT scéna": [
        "czech", "česká republika", "brno", "praha", "prague", "avast", "seznam",
        "mall", "rohlik", "alza", "kiwi", "productboard", "apiary",
    ],
    "Open Source": [
        "open source", "github", "open-source", "komunita", "community", "kontribuce",
        "contribution", "licence", "mit license", "apache",
    ],
}


def detect_topics(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(topic)
    return found if found else ["Ostatní"]


def extract_guest_name(title: str) -> str:
    """Extract guest name from title format: 'S14E06 - Guest Name - Topic'"""
    parts = [p.strip() for p in title.split(" - ")]
    if len(parts) >= 2:
        # Skip the episode code (S14E06 pattern)
        for part in parts:
            if not re.match(r"^S\d+E\d+$", part, re.IGNORECASE):
                return part
    return ""


def parse_duration(entry: Any) -> float:
    """Try to get duration in minutes from RSS entry."""
    # iTunes duration tag: can be "HH:MM:SS" or "MM:SS" or seconds
    itunes_duration = entry.get("itunes_duration", "")
    if itunes_duration:
        parts = str(itunes_duration).split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
            elif len(parts) == 2:
                return int(parts[0]) + int(parts[1]) / 60
            else:
                return int(itunes_duration) / 60
        except (ValueError, TypeError):
            pass
    return 0.0


def get_summary(entry: Any, transcript_path: Path | None) -> str:
    """Use RSS description, falling back to first sentences of transcript."""
    # Try RSS summary/description first
    summary = entry.get("summary", "") or entry.get("description", "")
    if summary:
        # Strip HTML tags
        summary = re.sub(r"<[^>]+>", "", summary).strip()
        if len(summary) > 20:
            return summary[:500]

    # Fall back to first 3 sentences of transcript
    if transcript_path and transcript_path.exists():
        text = transcript_path.read_text(encoding="utf-8")
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return " ".join(sentences[:3])

    return ""


def match_transcript(title: str, transcripts_dir: Path) -> str:
    """Find matching transcript file for an episode title."""
    if not transcripts_dir.exists():
        return ""

    # 1. Try SxxExx episode code match (e.g. s14e06)
    ep_code = re.search(r"(s\d+e\d+)", title.lower())
    if ep_code:
        code = ep_code.group(1)
        for f in transcripts_dir.glob("*.txt"):
            if code in f.stem.lower():
                return str(f)

    # 2. Fallback: keyword match from title words (for bonus/bloopers/specials)
    # Slugify title: lowercase, keep only alphanumeric, split to words
    title_words = re.sub(r"[^a-z0-9\s]", "", title.lower()).split()
    # Use first 3 meaningful words (skip short ones)
    keywords = [w for w in title_words if len(w) > 3][:3]
    if keywords:
        for f in transcripts_dir.glob("*.txt"):
            stem = f.stem.lower()
            if sum(1 for kw in keywords if kw in stem) >= 2:
                return str(f)

    return ""


def main() -> None:
    config = load_config()
    rss_url = config["podcast"]["rss_url"]
    transcripts_dir = Path(config["paths"]["transcripts_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "episode_metadata.json"

    print(f"📡 Načítám RSS feed: {rss_url}")
    feed = feedparser.parse(rss_url)
    entries = feed.entries
    print(f"   Nalezeno {len(entries)} epizod v RSS feedu.\n")

    episodes = []
    for i, entry in enumerate(entries, start=1):
        title = entry.get("title", f"Epizoda {i}")
        guest_name = extract_guest_name(title)
        date_parsed = entry.get("published_parsed")
        date_str = ""
        if date_parsed:
            import time
            date_str = time.strftime("%Y-%m-%d", date_parsed)

        duration = parse_duration(entry)
        transcript_file = match_transcript(title, transcripts_dir)
        transcript_path = Path(transcript_file) if transcript_file else None

        # Build text for topic detection: title + description + transcript snippet
        detect_text = title + " " + entry.get("summary", "")
        if transcript_path and transcript_path.exists():
            detect_text += " " + transcript_path.read_text(encoding="utf-8", errors="ignore")[:3000]

        topics = detect_topics(detect_text)
        summary = get_summary(entry, transcript_path)

        episode = {
            "episode_id": i,
            "title": title,
            "guest_name": guest_name,
            "date": date_str,
            "duration_minutes": round(duration, 1),
            "topics": topics,
            "summary": summary,
            "transcript_file": transcript_file,
        }
        episodes.append(episode)
        print(f"✅ [{i:03d}] {title[:60]}")
        if guest_name:
            print(f"       Host: {guest_name} | Témata: {', '.join(topics)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 episode_metadata.json uložen: {output_path}")
    print(f"   Celkem epizod: {len(episodes)}")


if __name__ == "__main__":
    main()
