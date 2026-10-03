"""Local meeting-minutes generator.
Upload recording -> faster-whisper transcript -> local LLM (Ollama) -> structured minutes.
Run:  streamlit run app.py
"""
import os

# Hard offline: block any Hugging Face network lookups (must be set before imports)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import datetime
import json
import shutil
import subprocess
import tempfile
import time

import requests
import streamlit as st
from faster_whisper import WhisperModel

# Only ever talk to a local Ollama server
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
assert OLLAMA_URL.startswith(("http://127.0.0.1", "http://localhost")), "OLLAMA_URL must be local"
CHUNK_CHARS = 12000  # transcripts longer than this are summarised in stages

SCHEMA_HINT = """Return ONLY valid JSON with this shape:
{
  "title": "short meeting title",
  "summary": "5-8 sentence executive summary",
  "attendees": ["names mentioned as participants"],
  "minutes": [{"topic": "...", "discussion": ["key point", "..."]}],
  "decisions": [{"decision": "...", "context": "why / who agreed"}],
  "action_items": [{"task": "...", "owner": "name or 'Unassigned'",
                    "due_date": "YYYY-MM-DD or 'Not specified'", "priority": "High|Medium|Low"}],
  "open_questions": ["unresolved items or risks"]
}
Rules: never invent facts, owners or dates that are not in the transcript. Resolve relative
dates ("next Friday") using the meeting date given. Use 'Unassigned' / 'Not specified' when unclear."""


@st.cache_resource(show_spinner=False)
def load_whisper(size: str) -> WhisperModel:
    return WhisperModel(size, device="auto", compute_type="auto", local_files_only=True)


def ts(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def to_wav(src: str) -> str:
    """Convert any audio/video to 16 kHz mono WAV with system ffmpeg (more tolerant than PyAV)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install it with:  brew install ffmpeg")
    dst = src + ".wav"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-err_detect", "ignore_err", "-i", src,
         "-vn", "-ac", "1", "-ar", "16000", dst],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 1000:
        raise RuntimeError(
            "ffmpeg could not read this file; it is likely corrupted or not real audio. "
            f"Details: {r.stderr.strip()[:300]}"
        )
    return dst


def transcribe(path: str, size: str, on_progress=None) -> str:
    segments, info = load_whisper(size).transcribe(path, vad_filter=True)
    lines = []
    for s in segments:
        lines.append(f"[{ts(s.start)}] {s.text.strip()}")
        if on_progress and info.duration:
            on_progress(min(s.end / info.duration, 1.0))
    return "\n".join(lines)


def check_ollama(model: str) -> None:
    """Fail fast with a clear message instead of hanging."""
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except Exception:
        raise RuntimeError("Ollama isn't running. Open the Ollama app (or run `ollama serve`) and retry.")
    names = [m["name"] for m in tags.get("models", [])]
    if model not in names and f"{model}:latest" not in names:
        raise RuntimeError(f"Model '{model}' is not installed. Run: ollama pull {model}   (installed: {names})")


def chat(model: str, system: str, user: str, as_json: bool, on_token=None, num_ctx: int = 8192) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": num_ctx},
    }
    if as_json:
        payload["format"] = "json"
    out = []
    # (connect timeout, max silence between streamed chunks)
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(10, 300)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            out.append(chunk.get("message", {}).get("content", ""))
            if on_token:
                on_token(len(out))
            if chunk.get("done"):
                break
    return "".join(out)


def chunks(text: str, size: int):
    buf = ""
    for line in text.splitlines():
        if len(buf) + len(line) > size and buf:
            yield buf
            buf = ""
        buf += line + "\n"
    if buf:
        yield buf


def extract(transcript: str, model: str, meeting_date: str, status) -> dict:
    system = "You are a precise meeting secretary. " + SCHEMA_HINT
    if len(transcript) > CHUNK_CHARS:
        notes = []
        parts = list(chunks(transcript, CHUNK_CHARS))
        for i, part in enumerate(parts, 1):
            status.update(label=f"Condensing part {i}/{len(parts)}...")
            notes.append(chat(
                model,
                "You are a meeting secretary. Write dense bullet notes of this transcript part: topics, "
                "decisions, action items with owner and any deadline wording, open questions, names. No invention.",
                part, as_json=False,
                on_token=lambda n, i=i: status.update(label=f"Step 2/2: condensing part {i}/{len(parts)} ({n} chunks)")))
        source = "\n\n".join(notes)
    else:
        source = transcript
    status.update(label="Step 2/2: waiting for the model to read the transcript...")
    raw = chat(
        model, system, f"Meeting date: {meeting_date}\n\nTRANSCRIPT/NOTES:\n{source}", as_json=True,
        on_token=lambda n: status.update(label=f"Step 2/2: writing minutes ({n} chunks generated)"),
    )
    return json.loads(raw)


def to_markdown(d: dict, meeting_date: str) -> str:
    out = [f"# {d.get('title', 'Meeting Minutes')}", f"*Date: {meeting_date}*", ""]
    if d.get("attendees"):
        out += ["**Attendees:** " + ", ".join(d["attendees"]), ""]
    out += ["## Summary", d.get("summary", ""), "", "## Minutes"]
    for m in d.get("minutes", []):
        out.append(f"### {m.get('topic', '')}")
        out += [f"- {p}" for p in m.get("discussion", [])]
    out += ["", "## Decisions"]
    out += [f"- **{x.get('decision', '')}** — {x.get('context', '')}" for x in d.get("decisions", [])]
    out += ["", "## Action Items", "| Task | Owner | Due | Priority |", "|---|---|---|---|"]
    for a in d.get("action_items", []):
        out.append(f"| {a.get('task','')} | {a.get('owner','')} | {a.get('due_date','')} | {a.get('priority','')} |")
    out += ["", "## Open Questions"] + [f"- {q}" for q in d.get("open_questions", [])]
    return "\n".join(out)


# ---------------- UI ----------------
st.set_page_config(page_title="Meeting Minutes", page_icon="📝", layout="wide")
st.title("📝 Meeting Minutes Generator")
st.caption("Everything runs locally: audio never leaves your machine.")

with st.sidebar:
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=1)
    llm_model = st.text_input("Ollama model", "llama3.1:8b")
    meeting_date = st.date_input("Meeting date", datetime.date.today()).isoformat()

file = st.file_uploader("Upload recording", type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "ogg", "flac"])

if file and st.button("Generate minutes", type="primary"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
        tmp.write(file.getbuffer())
        path = tmp.name
    wav = None
    try:
        check_ollama(llm_model)  # fail fast if Ollama is down or the model is missing
        with st.status("Step 1/2: transcribing...", expanded=True) as status:
            t0 = time.time()
            status.update(label="Step 1/2: converting audio and loading Whisper (first run is slower)...")
            wav = to_wav(path)
            transcript = transcribe(
                wav, whisper_size,
                on_progress=lambda p: status.update(label=f"Step 1/2: transcribing {p:.0%}"),
            )
            st.session_state["transcript"] = transcript
            st.write(f"Transcript ready in {time.time() - t0:.0f}s ({len(transcript)} characters)")

            t1 = time.time()
            st.session_state["result"] = extract(transcript, llm_model, meeting_date, status)
            st.session_state["date"] = meeting_date
            st.write(f"Minutes ready in {time.time() - t1:.0f}s")
            status.update(label="Done", state="complete")
    except Exception as e:
        st.error(f"Failed: {e}")
    finally:
        os.unlink(path)
        if wav and os.path.exists(wav):
            os.unlink(wav)

if "result" in st.session_state:
    d, date = st.session_state["result"], st.session_state["date"]
    tab1, tab2, tab3 = st.tabs(["Minutes", "Action items", "Transcript"])
    md = to_markdown(d, date)
    with tab1:
        st.markdown(md)
        c1, c2 = st.columns(2)
        c1.download_button("Download .md", md, "minutes.md")
        c2.download_button("Download .json", json.dumps(d, indent=2), "minutes.json")
    with tab2:
        st.data_editor(d.get("action_items", []), use_container_width=True, num_rows="dynamic")
    with tab3:
        st.text_area("Transcript", st.session_state["transcript"], height=500)