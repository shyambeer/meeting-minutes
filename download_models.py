"""Run ONCE while online to cache Whisper models. After this the app works fully offline.
Usage: python download_models.py small medium
"""
import sys

from faster_whisper import WhisperModel

sizes = sys.argv[1:] or ["small"]
for size in sizes:
    print(f"Downloading Whisper '{size}' ...")
    WhisperModel(size, device="cpu", compute_type="int8")  # downloads to the HF cache
    print(f"  cached: {size}")
print("Done. Now also run:  ollama pull llama3.1:8b")
