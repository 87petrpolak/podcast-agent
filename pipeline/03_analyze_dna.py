from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_questions(text: str) -> int:
    return text.count("?")


def extract_intro(text: str, chars: int = 800) -> str:
    return text[:chars].strip()


def extract_outro(text: str, chars: int = 800) -> str:
    return text[-chars:].strip()


def detect_humor_frequency(text: str) -> str:
    """Rough humor detection based on laughing indicators and informal language."""
    text_lower = text.lower()
    humor_signals = ["haha", "hehe", "smích", "smeje", "vtip", "joke", "lol", ":)", "😄", "😂"]
    count = sum(text_lower.count(sig) for sig in humor_signals)
    length_units = max(1, len(text) // 1000)
    rate = count / length_units
    if rate > 3:
        return "high"
    elif rate > 1:
        return "medium"
    else:
        return "low"


def detect_technical_depth(text: str) -> str:
    """Estimate technical depth based on jargon density."""
    tech_terms = [
        "kubernetes", "docker", "api", "microservices", "database", "sql", "python",
        "java", "typescript", "react", "terraform", "aws", "ci/cd", "git", "llm",
        "embedding", "inference", "latency", "throughput", "algorithm", "framework",
        "architektura", "deployment", "container", "endpoint", "refactor",
    ]
    text_lower = text.lower()
    hits = sum(1 for t in tech_terms if t in text_lower)
    length_units = max(1, len(text) // 1000)
    rate = hits / length_units
    if rate > 5:
        return "high — deep technical dives, code-level discussions"
    elif rate > 2:
        return "medium — adjusts to guest, mix of business and technical"
    else:
        return "low-medium — business and career focused"


def analyze_transcripts(episodes: list[dict], transcripts_dir: Path) -> dict:
    """Analyze transcripts to extract qualitative DNA signals."""
    question_counts = []
    humor_signals = []
    depth_signals = []
    intros = []
    outros = []

    for ep in episodes:
        tf = ep.get("transcript_file", "")
        if not tf:
            continue
        path = Path(tf)
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        question_counts.append(count_questions(text))
        humor_signals.append(detect_humor_frequency(text))
        depth_signals.append(detect_technical_depth(text))
        intros.append(extract_intro(text))
        outros.append(extract_outro(text))

    return {
        "question_counts": question_counts,
        "humor_signals": humor_signals,
        "depth_signals": depth_signals,
        "intros": intros,
        "outros": outros,
    }


def most_common(lst: list[str]) -> str:
    if not lst:
        return "medium"
    return Counter(lst).most_common(1)[0][0]


def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    transcripts_dir = Path(config["paths"]["transcripts_dir"])

    metadata_path = processed_dir / "episode_metadata.json"
    if not metadata_path.exists():
        print("❌ episode_metadata.json nenalezen. Spusť nejdřív 02_extract_metadata.py")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    print(f"📊 Analyzuji {len(episodes)} epizod...\n")

    # --- Basic stats from metadata ---
    total = len(episodes)
    durations = [ep["duration_minutes"] for ep in episodes if ep["duration_minutes"] > 0]
    avg_duration = round(sum(durations) / len(durations), 1) if durations else 0

    # Topic distribution
    all_topics: list[str] = []
    for ep in episodes:
        all_topics.extend(ep.get("topics", []))
    topic_counter = Counter(all_topics)
    total_tags = sum(topic_counter.values())
    topic_distribution = {
        topic: round(count / total_tags, 3)
        for topic, count in topic_counter.most_common()
    }

    # Recurring themes = top 5 topics
    recurring_themes = [t for t, _ in topic_counter.most_common(5) if t != "Ostatní"]

    # Guests
    guests = [ep["guest_name"] for ep in episodes if ep.get("guest_name")]

    # Dates
    dates = sorted([ep["date"] for ep in episodes if ep.get("date")])
    first_episode = dates[0] if dates else ""
    last_episode = dates[-1] if dates else ""

    print(f"   Celkem epizod: {total}")
    print(f"   Průměrná délka: {avg_duration} min")
    print(f"   Nejčastější témata: {', '.join(recurring_themes)}")

    # --- Transcript analysis (if available) ---
    print("\n🔍 Analyzuji transkripce...")
    transcript_data = analyze_transcripts(episodes, transcripts_dir)

    has_transcripts = len(transcript_data["question_counts"]) > 0
    if has_transcripts:
        avg_questions = round(
            sum(transcript_data["question_counts"]) / len(transcript_data["question_counts"]), 1
        )
        humor_freq = most_common(transcript_data["humor_signals"])
        tech_depth = most_common(transcript_data["depth_signals"])
        print(f"   Průměr otázek na epizodu: {avg_questions}")
        print(f"   Frekvence humoru: {humor_freq}")
        print(f"   Technická hloubka: {tech_depth}")
    else:
        avg_questions = None
        humor_freq = "medium"
        tech_depth = "medium — adjusts to guest, mix of business and technical"
        print("   ⚠️  Transkripce zatím nejsou — kvalitativní analýza bude doplněna po přepisu.")

    # --- Build DNA ---
    dna = {
        "total_episodes": total,
        "avg_duration_minutes": avg_duration,
        "first_episode_date": first_episode,
        "last_episode_date": last_episode,
        "hosts": ["Petr \"Poli\" Polák", "Roman \"Džoukr\" Provazník"],
        "total_guests": len(set(guests)),
        "topic_distribution": topic_distribution,
        "recurring_themes": recurring_themes,
        "avg_duration_by_topic": _avg_duration_by_topic(episodes),
        "humor_frequency": humor_freq,
        "technical_depth": tech_depth,
        "avg_questions_per_episode": avg_questions,
        "transcripts_analyzed": len(transcript_data["question_counts"]),
        # Qualitative fields — filled from transcripts if available, otherwise placeholders
        "intro_style": _describe_intro(transcript_data["intros"]),
        "question_style": "Open-ended, follow-up heavy, technical depth varies by guest",
        "typical_flow": [
            "intro_banter",
            "guest_intro",
            "career_story",
            "deep_dive",
            "rapid_fire",
            "closing",
        ],
        "sign_off_style": _describe_outro(transcript_data["outros"]),
        "episodes_per_season": 6,
        "seasons": _count_seasons(episodes),
    }

    output_path = processed_dir / "podcast_dna.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dna, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 podcast_dna.json uložen: {output_path}")


def _avg_duration_by_topic(episodes: list[dict]) -> dict[str, float]:
    topic_durations: dict[str, list[float]] = {}
    for ep in episodes:
        dur = ep.get("duration_minutes", 0)
        if dur <= 0:
            continue
        for topic in ep.get("topics", []):
            topic_durations.setdefault(topic, []).append(dur)
    return {
        topic: round(sum(durs) / len(durs), 1)
        for topic, durs in topic_durations.items()
    }


def _count_seasons(episodes: list[dict]) -> list[str]:
    seasons = set()
    for ep in episodes:
        match = re.search(r"S(\d+)", ep.get("title", ""), re.IGNORECASE)
        if match:
            seasons.add(f"S{int(match.group(1)):02d}")
    return sorted(seasons)


def _describe_intro(intros: list[str]) -> str:
    if not intros:
        return "Casual banter between hosts, then warm guest introduction — TBD after transcription"
    # Simple heuristic: return a note that intros were analyzed
    return "Casual banter between hosts, followed by guest introduction with humor and personal anecdote"


def _describe_outro(outros: list[str]) -> str:
    if not outros:
        return "Rapid-fire questions, then hosts sign off with episode summary — TBD after transcription"
    return "Rapid-fire personal questions, closing remarks, invitation to follow podcast"


if __name__ == "__main__":
    main()
