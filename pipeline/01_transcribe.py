from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from faster_whisper import WhisperModel


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def transcribe_file(model: WhisperModel, audio_path: Path, output_path: Path) -> None:
    print(f"🎙️  Přepisuji: {audio_path.name}")
    segments, info = model.transcribe(
        str(audio_path),
        language="cs",
        beam_size=5,
        vad_filter=True,
    )

    print(f"   Délka: {info.duration / 60:.1f} min | Jazyk: {info.language} ({info.language_probability:.0%})")

    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.text.strip() + "\n")

    print(f"✅ Uloženo: {output_path.name}")


def main() -> None:
    config = load_config()
    audio_dir = Path(config["paths"]["audio_dir"])
    transcripts_dir = Path(config["paths"]["transcripts_dir"])
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    print("⏳ Načítám model faster-whisper large-v3 (první spuštění stáhne ~3 GB)...")
    model = WhisperModel("large-v3", device="auto", compute_type="auto")
    print("✅ Model připraven.\n")

    audio_files = sorted(audio_dir.glob("*.mp3"))
    if not audio_files:
        print("❌ Žádné MP3 soubory v data/audio/")
        return

    print(f"Nalezeno {len(audio_files)} epizod ke zpracování.\n")

    for i, audio_path in enumerate(audio_files, start=1):
        output_path = transcripts_dir / (audio_path.stem + ".txt")

        if output_path.exists():
            print(f"⏭️  Přeskakuji (už existuje): {output_path.name}")
            continue

        print(f"[{i}/{len(audio_files)}] ", end="")
        try:
            transcribe_file(model, audio_path, output_path)
        except Exception as e:
            print(f"❌ Chyba u '{audio_path.name}': {e}")

        print()

    print("🎉 Hotovo! Všechny epizody přepsány.")


if __name__ == "__main__":
    main()
