# Local Meeting Minutes Generator (fully offline)

## One-time setup (needs internet)
1. Install Ollama (https://ollama.com) and pull a model:
   `ollama pull llama3.1:8b`   (`qwen2.5:14b` is better if you have ~16 GB RAM/VRAM)
2. Install Python deps (3.10+):
   `pip install streamlit faster-whisper requests`
3. Cache the Whisper model(s):
   `python download_models.py small`
   (for offline installs on another machine, copy `~/.cache/huggingface` and `~/.ollama`)

## Run (no internet needed)
`streamlit run app.py`  ->  opens at http://127.0.0.1:8501

## What keeps it offline
- `HF_HUB_OFFLINE=1` and `local_files_only=True`: Whisper loads only from the local cache.
- LLM calls are restricted to a localhost Ollama server (the app refuses non-local URLs).
- Streamlit telemetry is off and the server binds to 127.0.0.1 only.
- Uploaded audio is written to a temp file and deleted right after processing.

## Output
Summary, minutes by topic, decisions, action items (owner, due date, priority), open questions.
Export as Markdown or JSON; action items are editable in the UI.

## Next upgrades
- Speaker diarization (pyannote.audio, models can be cached for offline use).
- SQLite history and search across meetings.

## Launch
1. https://github.com/user-attachments/assets/2ac15f69-8a94-4ca4-9a70-446fcb56de46

## Output
https://github.com/user-attachments/assets/28576988-816a-43c2-beaf-cec5a41826b2

