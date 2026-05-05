from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])

    metadata_path = processed_dir / "episode_metadata.json"
    if not metadata_path.exists():
        print("❌ episode_metadata.json nenalezen. Spusť nejdřív 02_extract_metadata.py")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    print(f"🕸️  Stavím knowledge graph z {len(episodes)} epizod...\n")

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    topic_episode_count: dict[str, int] = defaultdict(int)
    guest_episodes: dict[str, list[int]] = defaultdict(list)

    for ep in episodes:
        ep_id = ep["episode_id"]
        title = ep["title"]
        guest = ep.get("guest_name", "").strip()
        topics = ep.get("topics", [])
        date = ep.get("date", "")

        # Episode node
        ep_node_id = f"ep_{ep_id}"
        nodes[ep_node_id] = {
            "id": ep_node_id,
            "label": title[:50],
            "type": "episode",
            "episode_id": ep_id,
            "date": date,
            "full_title": title,
        }

        # Guest node
        if guest and not _is_episode_code(guest):
            if guest not in nodes:
                nodes[guest] = {
                    "id": guest,
                    "label": guest,
                    "type": "guest",
                    "episode_count": 0,
                }
            nodes[guest]["episode_count"] = nodes[guest].get("episode_count", 0) + 1
            guest_episodes[guest].append(ep_id)

            # Edge: guest → episode
            edges.append({
                "source": guest,
                "target": ep_node_id,
                "type": "appeared_in",
                "episode_id": ep_id,
            })

        # Topic nodes + edges
        for topic in topics:
            if topic == "Ostatní":
                continue
            topic_episode_count[topic] += 1

            if topic not in nodes:
                nodes[topic] = {
                    "id": topic,
                    "label": topic,
                    "type": "topic",
                    "episode_count": 0,
                }
            nodes[topic]["episode_count"] = topic_episode_count[topic]

            # Edge: episode → topic
            edges.append({
                "source": ep_node_id,
                "target": topic,
                "type": "covers_topic",
                "episode_id": ep_id,
            })

            # Edge: guest → topic (if guest exists)
            if guest and not _is_episode_code(guest):
                edges.append({
                    "source": guest,
                    "target": topic,
                    "type": "expert_in",
                    "episode_id": ep_id,
                })

    # Topic gap analysis — topics with 0 episodes (from our taxonomy)
    all_taxonomy_topics = [
        "AI & ML", "Startupy", "Cloud & DevOps", "Architektura",
        "Kariéra", "Bezpečnost", "Data & Analytics", "Podnikání & Produkt",
        "Česká IT scéna", "Open Source",
    ]
    gaps = [t for t in all_taxonomy_topics if topic_episode_count.get(t, 0) == 0]

    graph = {
        "nodes": list(nodes.values()),
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_guests": sum(1 for n in nodes.values() if n["type"] == "guest"),
            "total_topics": sum(1 for n in nodes.values() if n["type"] == "topic"),
            "total_episodes": sum(1 for n in nodes.values() if n["type"] == "episode"),
            "topic_coverage": {t: topic_episode_count[t] for t in all_taxonomy_topics},
            "topic_gaps": gaps,
            "most_covered_topics": sorted(
                topic_episode_count.items(), key=lambda x: x[1], reverse=True
            )[:5],
        },
    }

    output_path = processed_dir / "topic_graph.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"   Uzly celkem: {graph['stats']['total_nodes']}")
    print(f"   Hosti: {graph['stats']['total_guests']}")
    print(f"   Témata: {graph['stats']['total_topics']}")
    print(f"   Hrany: {graph['stats']['total_edges']}")
    if gaps:
        print(f"\n⚠️  Témata, která podcast zatím nepokryl: {', '.join(gaps)}")
    else:
        print("\n✅ Všechna témata z taxonomie jsou pokryta.")

    print(f"\n🎉 topic_graph.json uložen: {output_path}")


def _is_episode_code(text: str) -> bool:
    """Detect false guest names like 'S07' or 'BonusOFFka #1'."""
    import re
    return bool(re.match(r"^S\d+$", text)) or "Bonus" in text or "#" in text


if __name__ == "__main__":
    main()
