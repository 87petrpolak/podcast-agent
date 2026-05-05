from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_episode_text(ep: dict) -> str:
    """Build text representation of episode for embedding."""
    parts = [ep.get("title", ""), ep.get("guest_name", ""), ep.get("summary", "")]
    parts.extend(ep.get("topics", []))

    # Add transcript snippet if available
    tf = ep.get("transcript_file", "")
    if tf:
        path = Path(tf)
        if path.exists():
            transcript = path.read_text(encoding="utf-8", errors="ignore")
            parts.append(transcript[:2000])

    return " ".join(filter(None, parts))


def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])

    metadata_path = processed_dir / "episode_metadata.json"
    if not metadata_path.exists():
        print("❌ episode_metadata.json nenalezen. Spusť nejdřív 02_extract_metadata.py")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    print(f"🔢 Počítám podobnost mezi {len(episodes)} epizodami...")
    print("   Načítám embedding model (první spuštění stáhne ~500 MB)...\n")

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("   ✅ Model připraven.\n")

    # Build texts
    texts = []
    episode_ids = []
    episode_titles = []
    for ep in episodes:
        text = get_episode_text(ep)
        texts.append(text)
        episode_ids.append(ep["episode_id"])
        episode_titles.append(ep["title"])

    print(f"   Embeduji {len(texts)} epizod...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)

    print("\n   Počítám cosine similarity matici...")
    sim_matrix = cosine_similarity(embeddings)

    # Round to 3 decimal places to keep file size reasonable
    sim_matrix_rounded = np.round(sim_matrix, 3).tolist()

    # Find top similar pairs (excluding self-similarity)
    top_pairs = []
    n = len(episode_ids)
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix_rounded[i][j]
            if score > 0.7:
                top_pairs.append({
                    "episode_a_id": episode_ids[i],
                    "episode_a_title": episode_titles[i],
                    "episode_b_id": episode_ids[j],
                    "episode_b_title": episode_titles[j],
                    "similarity": score,
                })

    top_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    output = {
        "episode_ids": episode_ids,
        "episode_titles": episode_titles,
        "matrix": sim_matrix_rounded,
        "high_similarity_pairs": top_pairs[:20],  # top 20 most similar pairs
        "stats": {
            "total_episodes": n,
            "pairs_above_70pct": len(top_pairs),
            "avg_similarity": round(float(np.mean(sim_matrix[~np.eye(n, dtype=bool)])), 3),
        },
    }

    output_path = processed_dir / "similarity_matrix.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Statistiky:")
    print(f"   Průměrná podobnost: {output['stats']['avg_similarity']:.1%}")
    print(f"   Páry s >70% podobností: {output['stats']['pairs_above_70pct']}")
    if top_pairs:
        print(f"\n   Top 3 nejpodobnější páry:")
        for pair in top_pairs[:3]:
            print(f"   {pair['similarity']:.0%} — {pair['episode_a_title'][:40]} ↔ {pair['episode_b_title'][:40]}")

    print(f"\n🎉 similarity_matrix.json uložen: {output_path}")


if __name__ == "__main__":
    main()
