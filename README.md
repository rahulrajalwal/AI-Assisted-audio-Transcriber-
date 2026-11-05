# AI Meeting Assistant

Turn meeting recordings into searchable knowledge. Give it a **YouTube link** or a **local audio/video file**. It transcribes the speech, writes a structured summary, extracts **action items, key decisions and open questions**, and lets you **ask questions** answered only from what was actually said, with sources.

## Main features

- **Inputs:** YouTube URLs, or uploaded audio/video files (mp3, wav, m4a, flac, ogg, mp4, mkv, mov, webm, …).
- **Two transcription engines:** OpenAI Whisper runs locally for English. Sarvam AI handles Hinglish (and returns English text).
- **Gemini 3.8 Flash analysis:**
  - Map-Reduce summarisation that scales to long recordings.
  - A short meeting title.
  - Three separate extraction chains (action items, key decisions, open questions) with strict no-fabrication rules.
- **RAG Q&A:** Hugging Face `all-MiniLM-L6-v2` embeddings stored persistently in ChromaDB, queried through a LangChain LCEL retrieval chain.
  - Answers cite `[Source N]` excerpts.
  - Follow-up questions are supported.
  - You can search one meeting or all of them.
- **Meeting history:** every processed meeting is saved and can be reopened. Re-processing the same recording reuses the saved analysis (no extra Gemini calls) and never duplicates vector records.
- **Exports:** transcript (TXT), summary with extractions (TXT), and a formatted PDF report (fpdf2).
- **Reliability:**
  - Real per-stage progress.
  - Bounded exponential-backoff retries for 429/5xx/timeouts that honour the server's retry delay.
  - Fail-fast on authentication, model and daily-quota errors.
  - No "success" is ever shown after a failed stage.

## Architecture and workflow

```text
YouTube URL / audio / video
  └─ 1. Input validation & audio extraction      utils/audio_processor.py  (yt-dlp)
  └─ 2. Conversion (mono, 16 kHz) & 10-min chunks utils/audio_processor.py  (FFmpeg + pydub)
  └─ 3. Speech-to-text                            core/transcriber.py       (Whisper | Sarvam AI)
  └─ 4. Map-Reduce summary + title                core/summarizer.py        (Gemini via LCEL)
  └─ 5. Action items / decisions / questions      core/extractor.py         (Gemini via LCEL)
  └─ 6. Embeddings → ChromaDB, RAG chain          core/vector_store.py, core/rag_engine.py
         → Streamlit UI (app.py) · CLI (main.py) · exports (utils/exporter.py)
```

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface: home, live progress, results tabs, chat, exports, history |
| `main.py` | Command-line interface (same pipeline) |
| `core/config.py` | Loads `.env` first; project paths |
| `core/llm.py` | **Single** Gemini setup (`ChatGoogleGenerativeAI`), response checks, retries, error messages |
| `core/pipeline.py` | The six-stage pipeline with progress events (shared by UI and CLI) |
| `core/summarizer.py` | Map-Reduce summary with bounded multi-level reduction; title generation |
| `core/extractor.py` | Three extraction chains; long transcripts are sectioned then consolidated |
| `core/rag_engine.py` | LCEL RAG: question → retriever → context → prompt → Gemini → answer + sources |
| `core/vector_store.py` | MiniLM embeddings, persistent ChromaDB, per-meeting metadata, de-duplication |
| `core/meeting_store.py` | Meeting history (one JSON file per meeting in `data/meetings/`) |
| `core/transcriber.py` | Whisper (English) and Sarvam AI (Hinglish) |
| `utils/audio_processor.py` | yt-dlp download, validation, FFmpeg/pydub conversion and chunking |
| `utils/exporter.py` | TXT and PDF exports |
| `tests/` | Offline test suite (mocks for Gemini, Sarvam and YouTube) |

## Technology stack

Python · Streamlit · OpenAI Whisper (local) · Sarvam AI · yt-dlp · FFmpeg + pydub · **Google Gemini 3.8 Flash** (`gemini-3.8-flash`) · LangChain + LCEL · Hugging Face `all-MiniLM-L6-v2` · ChromaDB · fpdf2

## Setup (Windows / PowerShell)

### 1. Python environment

Requires Python 3.10 or newer (tested with 3.11).

```powershell
cd "C:\path\to\AI Meeting Assistant"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

(With uv: `uv pip install -r requirements.txt`.) The first English run downloads the Whisper model. The first indexing run downloads `all-MiniLM-L6-v2` (about 90 MB).

### 3. Install FFmpeg

FFmpeg must be on your `PATH` (used by yt-dlp, pydub and Whisper):

```powershell
winget install Gyan.FFmpeg
```

Then close and reopen the terminal and check with `ffmpeg -version`.

### 4. Configure API keys

Copy the template and edit it:

```powershell
Copy-Item .env.example .env
notepad .env
```

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | yes | Gemini API key from https://aistudio.google.com/apikey (`GEMINI_API_KEY` is also accepted) |
| `GEMINI_MODEL` | no | Defaults to `gemini-3.8-flash` |
| `GEMINI_THINKING_LEVEL` | no | `low`, `medium` or `high` (`minimal` is not supported by 3.8 Flash) |
| `WHISPER_MODEL` | no | `tiny`, `base`, `small` (default), `medium`, `large`; larger models are more accurate but slower |
| `SARVAM_API_KEY` | for Hinglish | Sarvam AI subscription key |
| `SARVAM_STT_MODEL` | no | Defaults to `saaras:v2.5` |

Notes:

- Gemini 3 models ignore and deprecate `temperature`, `top_p` and `top_k`, so the app never sends them.
- Map steps and titles use thinking level `low` to save quota.
- The app never prints or displays key values. The sidebar shows only whether each key is configured.

## Running

```powershell
streamlit run app.py
```

Then open http://localhost:8501:

1. Choose **YouTube link** or **Upload file**.
2. Pick **English (Whisper)** or **Hinglish (Sarvam AI)**.
3. Click **Analyse meeting**.

When processing finishes, the report opens with tabs for Overview, Transcript, Summary, Action Items, Key Decisions, Open Questions and **Ask Your Meeting**.

Command-line alternative:

```powershell
python main.py "https://youtu.be/<video-id>" english
python main.py "C:\recordings\standup.mp4" hinglish
```

Tests (offline, no API calls; they use temporary folders, so your data is untouched):

```powershell
python -m pytest
```

`python test.py <source> <language>` runs a manual end-to-end smoke test against the **real** services.

## Data and storage

| Location | Contents |
|---|---|
| `vector_db/` | Persistent ChromaDB collection `meeting_transcript` (existing data is never cleared) |
| `data/meetings/` | One JSON record per processed meeting (transcript and analysis) |
| `downloads/` | Temporary audio, deleted automatically after each run |

All three are git-ignored. Back up `vector_db/` and `data/` if you want to keep your history.

## Gemini quotas and rate limits

Gemini 3.8 Flash accepts up to 1,048,576 input tokens, but **throughput is limited per Google Cloud project**. Free-tier limits are shown in AI Studio (https://aistudio.google.com/rate-limit) and are not guaranteed.

A typical meeting under about 75 minutes uses **5 Gemini calls**: summary, title and three extractions. Longer meetings add a few map/reduce calls, and each chat question uses one call.

The app handles limits as follows:

- **Rate limits (429):** retries with backoff, honouring Google's `retryDelay`.
- **Daily quota exhausted:** stops immediately with a clear message.
- **High demand (503):** retries, then asks you to try again later.

If transcription succeeded but analysis failed, use **Retry analysis**: the transcript is reused, so the audio isn't transcribed again. To spread requests out on a low RPM limit, set `GEMINI_MIN_REQUEST_INTERVAL` (for example, `6` for 10 requests per minute).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Gemini API key not found` | Add `GOOGLE_API_KEY` to `.env` in the project folder and restart Streamlit |
| `HTTP 401/403` from Gemini | The key is wrong, restricted, or belongs to another project; create a new key in AI Studio |
| `HTTP 404 … model not available` | Check `GEMINI_MODEL`; the model must be enabled for your project |
| `HTTP 503 … high demand` | Temporary overload at Google; wait a few minutes. To keep working meanwhile, you may set `GEMINI_MODEL=gemini-3.6-flash` (the closest previous-generation stable Flash) |
| `daily quota … exhausted (429)` | Free-tier daily requests are used up; they reset at midnight Pacific time, or enable billing |
| `FFmpeg was not found on PATH` | Install FFmpeg (step 3) and open a new terminal |
| YouTube download fails | Update yt-dlp: `python -m pip install -U yt-dlp`; private, age-restricted or region-locked videos can't be downloaded |
| `SARVAM_API_KEY is not set` | Needed only for Hinglish; add it to `.env` |
| Whisper is slow | Use `WHISPER_MODEL=base` or `tiny`, or a CUDA GPU (FP16 is enabled automatically) |
| `No speech was detected` | Check that the recording has audible speech and that the right language is selected |
| Upload too large | The limit is 1 GB (`.streamlit/config.toml` → `maxUploadSize`) |
| Hugging Face symlink warning on Windows | Harmless; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` or enable Developer Mode |
