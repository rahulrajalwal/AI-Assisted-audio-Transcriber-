# AI Meeting Assistant

Turn meeting recordings into searchable knowledge. Give it a **YouTube link** or a **local audio/video file**. It transcribes the speech, writes a structured summary, extracts **action items, key decisions and open questions**, and lets you **ask questions** about one meeting or your whole meeting history. Answers come only from what was actually said and cite their sources.

LLM processing runs on **OpenRouter** with **OpenAI GPT-4o mini** (`openai/gpt-4o-mini`). Transcription (Whisper and Sarvam AI) and embeddings (Hugging Face) are unchanged. The earlier Google Gemini and Mistral integrations are preserved, inactive, in `core/llm.py`.

## Main features

- **Inputs:** YouTube URLs, or uploaded audio/video files (mp3, wav, m4a, flac, ogg, mp4, mkv, mov, webm, …).
- **Transcription:** OpenAI Whisper runs locally for English. Sarvam AI handles Hindi/Hinglish (and returns English text). Both keep segment timestamps.
- **LLM analysis (OpenRouter, `openai/gpt-4o-mini`):**
  - Map-Reduce summarisation for long meetings.
  - A short meeting title.
  - Three separate extraction chains (action items, key decisions, open questions) with strict no-fabrication rules.
- **Persistent multi-meeting RAG:**
  - Transcripts are split into text chunks, embedded **locally** with Hugging Face `all-MiniLM-L6-v2` and stored in ChromaDB with meeting metadata.
  - Ask about one meeting or all of them, including time-aware questions such as "what did we decide **last week**?".
  - Answers cite `[Source N]` with the meeting, date and approximate time.
- **Interactive chatbot ("Ask Your Meeting"):**
  - Answers appear word by word as they're generated, and each one shows the transcript excerpts it used, with timestamps.
  - Clickable suggested questions, including ones built from the meeting's own decisions and action items.
  - Follow-up questions understand earlier ones (e.g. "And by when?").
  - Each answer has a copy button and 👍/👎 feedback. The chat can be downloaded or cleared.
  - **Chat history is saved with the meeting** (`data/meetings/chats/`), so it survives refreshes and restarts.
- **Timestamped transcript:** readable paragraphs with `[mm:ss]` marks taken from Whisper/Sarvam segments, plus search with highlighting.
- **Downloads tab:** every part on its own. That's the full PDF report (with the Q&A), a ZIP of everything, the transcript, summary, action items, key decisions, open questions and the chat.
- **Meeting history:** every processed meeting is saved and can be reopened. Re-processing the same recording reuses the saved analysis (no extra LLM requests) and never duplicates vector records.
- **Exports:** transcript (TXT), summary with extractions (TXT), and a formatted PDF report.
- **Cost- and limit-aware:**
  - Requests are sequential, with at most one retry; a meeting costs well under $0.01 with GPT-4o mini.
  - The app checks the key and model (and, for `:free` models, the daily quota) before starting.
  - It never switches models on its own.

## Architecture and workflow

```text
MEETING PROCESSING
  YouTube URL / audio / video
   └─ 1. Input validation & audio extraction        utils/audio_processor.py  (yt-dlp)
   └─ 2. Mono 16 kHz WAV + AUDIO chunks (10 min)     utils/audio_processor.py  (FFmpeg + pydub)
   └─ 3. Speech-to-text → complete transcript        core/transcriber.py       (Whisper | Sarvam AI)
   └─ 4. Map-Reduce summary + title                  core/summarizer.py        ┐ LangChain LCEL →
   └─ 5. Action items / decisions / open questions   core/extractor.py         ┘ OpenRouter (GPT-4o mini)
   └─ 6. TEXT chunks → Hugging Face embeddings → persistent ChromaDB (text + vectors + metadata)
                                                     core/vector_store.py

USER QUESTION                                        core/rag_engine.py
   question → all-MiniLM-L6-v2 query vector → ChromaDB similarity search
   → most relevant transcript chunks (text) → RAG prompt → OpenRouter (GPT-4o mini)
   → grounded answer with sources → Streamlit (app.py)
```

Only transcript text goes to OpenRouter. Audio files and embeddings never leave your machine, and the query vector is used only by ChromaDB. The LLM receives the retrieved text passages.

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface: home, live progress, results tabs, chat, exports, history, OpenRouter status |
| `main.py` | Command-line interface (same pipeline) |
| `core/config.py` | Loads `.env` first; project paths |
| `core/llm.py` | **Single** LLM setup (OpenRouter via `ChatOpenAI`), status/quota check, rate-limit handling; **inactive Gemini/Mistral code preserved here** |
| `core/pipeline.py` | The six-stage pipeline with progress events |
| `core/summarizer.py` | Map-Reduce summary; title generation |
| `core/extractor.py` | Three extraction chains |
| `core/rag_engine.py` | LCEL RAG: retriever → context → prompt → LLM → answer + sources |
| `core/vector_store.py` | Text chunking, MiniLM embeddings, persistent ChromaDB, metadata, upsert/dedup, backup |
| `core/meeting_store.py` | Meeting history (one JSON file per meeting in `data/meetings/`) |
| `core/transcriber.py` | Whisper (English) and Sarvam AI (Hindi/Hinglish) |
| `utils/audio_processor.py` | yt-dlp download, validation, FFmpeg/pydub conversion and audio chunking |
| `utils/exporter.py` | TXT and PDF exports |
| `scripts/eval_chunking.py` | Compare text-chunking settings for retrieval quality |
| `scripts/validate_storage_workflow.py` | Live end-to-end storage + RAG check in temporary folders (about 15 requests) |
| `tests/` | Offline test suite (OpenRouter, Sarvam and YouTube are mocked) |

## Technology stack

Python · Streamlit · OpenAI Whisper (local) · Sarvam AI · yt-dlp · FFmpeg + pydub · **OpenRouter (OpenAI GPT-4o mini)** · LangChain + LCEL (`langchain-openai`) · Hugging Face `all-MiniLM-L6-v2` · ChromaDB · fpdf2

## Setup (Windows / PowerShell)

### 1. Python environment

Requires Python 3.10 or newer (tested with 3.11). Using the existing virtual environment:

```powershell
cd "C:\path\to\AI Meeting Assistant"
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Without uv, run `python -m pip install -r requirements.txt` instead. For a fresh environment, run `python -m venv .venv` first. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### 2. Install FFmpeg

```powershell
winget install Gyan.FFmpeg
```

Close and reopen the terminal, then check with `ffmpeg -version`.

### 3. Get an OpenRouter API key

1. Create a key at https://openrouter.ai/settings/keys.
2. Add a few dollars of credits at https://openrouter.ai/settings/credits (GPT-4o mini is a paid model; a meeting costs well under $0.01).

### 4. Configure `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required for title, summary, extraction and Q&A |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | The model used for every LLM request. Any OpenRouter model ID works. Optional free alternatives: `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`. Changes apply after **Refresh status**; the app never switches models on its own |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large`; larger models are more accurate but slower |
| `SARVAM_API_KEY` | — | Needed only for Hindi/Hinglish |
| `SARVAM_STT_MODEL` | `saaras:v2.5` | Sarvam speech-to-text model |

The app never prints or displays key values. The sidebar shows only whether the key is valid.

## Running

```powershell
streamlit run app.py
```

Then open http://localhost:8501:

1. Choose **YouTube link** or **Upload file**.
2. Pick **English (Whisper)** or **Hinglish (Sarvam AI)**. Optionally set the **meeting date** under *Advanced options*.
3. Click **Analyse meeting**.

The sidebar shows:
- the provider (**OpenRouter**) and the model;
- whether the API key is valid and the model is available;
- how many free requests are left today (shown only for `:free` models).

The first page load takes a while as Python libraries load.

Command line:

```powershell
python main.py "C:\recordings\standup.mp4" english
```

## Model, cost and limits

**Default model: OpenAI GPT-4o mini (`openai/gpt-4o-mini`) via OpenRouter.**

| Property | Value |
|---|---|
| Context window | 128K tokens; the app sends at most ~24K per request |
| Max output | 16K tokens; the app requests at most 4,096 |
| Price | about **$0.15 per million input tokens** and **$0.60 per million output tokens** |
| Providers | OpenAI and Azure (paid capacity, so no shared free-model throttling) |
| Requires | OpenRouter credits: https://openrouter.ai/settings/credits |

How many requests a meeting uses:

| Action | Requests | Typical cost with GPT-4o mini |
|---|---|---|
| Processing a meeting under about 1.5 hours (transcript up to ~80,000 characters) | **5**: summary, title, and the three extractions | well under $0.01 |
| Longer meetings | a few more (Map-Reduce map and reduce steps) | about $0.01 |
| Each chat question | **1** | a fraction of a cent |
| Re-opening or re-processing a saved meeting | **0** | free |

How the app handles errors and limits:
- Requests run **one at a time**. A temporary 429 or server error is retried **once**, honouring `Retry-After`.
- Missing credits (HTTP 402), an invalid key or an unavailable model stop immediately with a clear message.
- **The app never switches to another model by itself.**
- If the LLM fails after transcription, the transcript is kept. Use **Retry analysis**; the audio is not transcribed again.

**Changing the model:** edit `OPENROUTER_MODEL` in `.env`, then click **Refresh status** in the sidebar. The app re-reads `.env` before every analysis, so no restart is needed.

**Free alternative (optional, set manually):** `google/gemma-4-31b-it:free` or `google/gemma-4-26b-a4b-it:free`.
- Free models are limited to 20 requests per minute and 50 per day (1,000 per day after buying ≥ 10 credits); the count resets at 00:00 UTC.
- With a free model, the app spaces requests 3.1 s apart and shows "Free requests today" in the sidebar.
- In testing, both Gemma 4 free models were repeatedly "temporarily rate-limited upstream" by Google AI Studio, their only provider. That's why the default is now GPT-4o mini.

## Text chunking, embeddings and storage

The project uses two separate chunking steps:

- **Audio chunks:** 10-minute WAV pieces (`utils/audio_processor.py`), so Whisper and Sarvam can process long recordings. ChromaDB never stores audio.
- **Text chunks:** the finished transcript is split by `RecursiveCharacterTextSplitter` into **~800-character, sentence-aware chunks with 120 characters of overlap** (`core/vector_store.py`). These are embedded locally with `all-MiniLM-L6-v2`, and **both the chunk text and its vector** are stored in ChromaDB.

Questions are embedded with the **same** model, and ChromaDB returns the most similar chunks. Only their text is passed to the LLM.

Each chunk is stored with:

- `meeting_id`, `title`, `meeting_date`, `meeting_ts`, `processed_at`
- `source`, `source_type` (youtube, upload or local file), `language`
- `chunk_id`, `chunk_index`, `char_start`, `char_end`
- `start_seconds`, `end_seconds`, `audio_chunk` (approximate audio position, when timestamps are available)
- `embedding_model`, `chunk_size`, `schema_version`

The full transcript is also kept in the meeting's JSON record in `data/meetings/`.

How the data is managed:

- **Everything persists** in `vector_db/` across restarts. The collection is never cleared or recreated.
- **No duplicates:** reprocessing the same recording is detected (stable meeting ID from the transcript) and skipped. *Re-run the AI analysis* in Advanced options replaces only that meeting's chunks (upsert by ID, then delete that meeting's leftover chunks).
- **Older data still works:** chunks written by older versions without metadata stay searchable in "All meetings" mode. No migration is required.
- **Backup:**

  ```powershell
  python -m core.vector_store backup
  ```

  This copies the database to `backups/vector_db-<timestamp>`. To roll back, stop the app and copy the backup folder back to `vector_db/`. To list what is stored:

  ```powershell
  python -m core.vector_store stats
  ```

**Historical questions:** in the chat's **All meetings** scope:

- "previous / last meeting" resolves to the meeting before the one you're viewing.
- "yesterday", "today", "this week", "last week" (the previous calendar week), "this month" and "last month" filter by meeting date.
- If no meeting matches, the app says so instead of guessing.

Limitation: there's no tracking of whether action items are done, so "which action items are still pending" can only be answered from what the transcripts say.

**Text-chunking evaluation** (`scripts/eval_chunking.py`, 12 questions over two meetings, real MiniLM embeddings):

| Setting | Chunks | Top-1 hit | Top-3 hit | Chunks ending on a sentence boundary |
|---|---|---|---|---|
| 500/50 (old) | 25 | 11/12 | 12/12 | 62% |
| **800/120, sentence-aware (current)** | 18 | 11/12 | 12/12 | **100%** |

## Running the tests

```powershell
python -m pytest
```

The tests run offline with OpenRouter, Sarvam and YouTube mocked. They use temporary folders, so your data is untouched, and they spend no API requests.

One live test sends exactly **one** real request:

```powershell
$env:LIVE_OPENROUTER="1"; python -m pytest -m openrouter
```

The full live storage and RAG check uses about 15 requests:

```powershell
python scripts/validate_storage_workflow.py
```

## Re-enabling Gemini or Mistral

The earlier hosted integrations are preserved, commented out, at the bottom of `core/llm.py` under **INACTIVE HOSTED-MODEL INTEGRATIONS**, with step-by-step instructions. They are never used as fallbacks.

## Troubleshooting

| Problem | Fix |
|---|---|
| `OpenRouter API key not found` | Add `OPENROUTER_API_KEY` to `.env` in the project folder and restart the app |
| `OpenRouter rejected the API key (HTTP 401)` | Create a new key at https://openrouter.ai/settings/keys |
| `insufficient credits (HTTP 402)` | Add credits at https://openrouter.ai/settings/credits, then use **Retry analysis** |
| The sidebar still shows the old model after editing `.env` | Click **Refresh status** (or start a new analysis); both re-read `.env`. Also close any older copies of the app still running in other terminals |
| `free OpenRouter request limit for today has been reached` | Only with `:free` models: wait for the reset at 00:00 UTC (05:30 IST), or switch to the paid default |
| `rate-limited right now (HTTP 429)` or `rate-limited by its provider` | Wait a minute and retry. With `:free` models this is common; the paid default avoids it |
| `Model … is not available on OpenRouter` | Check `OPENROUTER_MODEL`; free models are sometimes removed. You may choose another model yourself, but the app never switches automatically |
| `Could not reach OpenRouter` | Check your internet connection, then click **Refresh status** |
| `FFmpeg was not found on PATH` | Install FFmpeg and open a new terminal |
| YouTube download fails | `python -m pip install -U yt-dlp`; private or region-locked videos can't be downloaded |
| `SARVAM_API_KEY is not set` | Needed only for Hindi/Hinglish; add it to `.env` |
| Whisper is slow | Use `WHISPER_MODEL=base` |
| `ModuleNotFoundError` when starting the app | The `.venv` folder was moved, so its `.exe` launchers point to the old location. Use `python -m streamlit run app.py`, or repair with `python -m pip install --force-reinstall --no-deps streamlit==1.66.1` |
| `WinError 1450` in the Streamlit log | Already handled by `.streamlit/config.toml` (`folderWatchBlacklist`) |
