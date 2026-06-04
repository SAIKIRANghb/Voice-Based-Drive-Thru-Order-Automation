# Drive-Thru Voice Agent

<<<<<<< HEAD
An interactive drive-thru ordering demo that combines browser audio capture, acoustic echo cancellation, voice activity detection, local Whisper ASR, a LangGraph/Gemini ordering agent, Gemini TTS, and optional speaker analytics.

The app serves a browser dashboard where a simulated car starts a lane session. Customers can speak through the microphone or type a transcript. The backend processes each utterance, updates the order cart, and returns text plus optional synthesized speech.

## Features

- Flask and Socket.IO realtime web app.
- Browser microphone streaming as PCM-16 mono audio at 16 kHz.
- NLMS acoustic echo cancellation with live DSP level visualization.
- Silero VAD by default, with an optional energy-envelope fallback.
- Local `faster-whisper` ASR.
- LangGraph finite-state ordering flow backed by Gemini.
- Tool-calling order logic for inventory, pricing, cart updates, and promo codes.
- Hallucination guard that intercepts off-menu food items before TTS.
- Gemini TTS with PCM output buffered back into the echo-cancellation reference path.
- Optional live and offline speaker diarization with SpeechBrain ECAPA embeddings.

## Project Structure

```text
.
|-- app.py                         # Flask app, Socket.IO handlers, audio/session pipeline
|-- run.py                         # Recommended development runner with model preloading
|-- requirements.txt               # Python dependencies
|-- .env.example                   # Environment variable template
|-- static/
|   |-- index.html                 # Dashboard UI
|   |-- css/style.css              # Dashboard styling
|   `-- js/app.js                  # Socket.IO client, mic capture, PCM playback, UI updates
|-- voice_agent/
|   |-- config.py                  # Audio constants, menu registry, prompts, env helpers
|   |-- asr/
|   |   |-- base.py
|   |   `-- whisper.py             # faster-whisper adapter and model cache setup
|   |-- dsp/
|   |   |-- aec.py                 # NLMS adaptive echo cancellation
|   |   `-- vad.py                 # Silero/energy VAD gate
|   |-- llm/
|   |   |-- graph.py               # LangGraph state machine
|   |   |-- nodes.py               # Node implementations and Gemini calls
|   |   |-- tools.py               # Menu/order tools exposed to the LLM
|   |   `-- guardrails.py          # Off-menu response guard
|   |-- spkrec/
|   |   |-- embedding.py           # SpeechBrain speaker embeddings
|   |   `-- diarization.py         # Utterance clustering and speaker analytics
|   |-- tts/
|   |   |-- base.py
|   |   `-- gemini.py              # Gemini TTS adapter
|   `-- utils/
|       |-- circular_buffer.py     # Speaker reference buffer for AEC
|       `-- verify_diarization.py  # Speaker analytics helper
`-- tests/
    |-- test_config.py
    |-- test_nodes.py
    `-- test_vad.py
```

Runtime artifacts such as `cache/`, `__pycache__/`, `server.*.log`, and `crash.log` are generated locally and are not part of the core source architecture.

## Architecture

```text
Browser Dashboard
  |  start_session / send_text / audio_frame / update_settings
  v
Flask + Socket.IO (app.py)
  |
  |-- Per-client session store
  |     |-- LangGraph state
  |     |-- NLMS filter
  |     |-- VAD gate
  |     |-- circular speaker-reference buffer
  |     `-- utterance/speaker analytics buffers
  |
  |-- Audio path
  |     Browser mic PCM
  |       -> optional echo injection
  |       -> NLMS acoustic echo cancellation
  |       -> VAD gate
  |       -> utterance buffering
  |       -> faster-whisper ASR
  |
  |-- Agent path
  |     transcript
  |       -> LangGraph FSM
  |       -> Gemini LLM
  |       -> order tools
  |       -> hallucination guard
  |
  `-- Output path
        response text
          -> Gemini TTS
          -> PCM audio response
          -> circular reference buffer for future AEC
          -> browser playback
```

## Conversation State Machine

The ordering agent is implemented in `voice_agent/llm/graph.py` as a LangGraph finite-state machine.

```text
START
  -> greeting
  -> taking_order <----+
  -> confirming        |
  -> upsell            |
  -> closing           |
  -> END               |
       ^               |
       +-- correction -+
```

The active state is stored per Socket.IO connection. Each turn carries:

- `messages`: recent user and assistant messages.
- `cart`: menu item quantities.
- `total_price`: calculated order total.
- `current_node` and `next_node`: FSM routing state.
- `discount` and `free_items`: promo state.
- `last_response`: latest customer-facing answer.
- `hallucination_warning`: whether the guardrail intercepted text.
- `state_timeout_ms`: UI/runtime timeout hint for the active state.

## Core Data Flow

1. The browser connects through Socket.IO and asks the user to drive the car to the speaker.
2. `start_session` initializes a per-client session and runs the initial greeting turn.
3. If microphone mode is enabled, the browser captures audio, resamples it to 16 kHz PCM-16, and streams `audio_frame` packets.
4. The backend optionally simulates speaker echo, runs NLMS echo cancellation, emits DSP levels, and applies VAD.
5. When speech ends, buffered audio is transcribed by `WhisperASR`.
6. The transcript is sent to `run_agent_turn`.
7. LangGraph routes the turn to the appropriate node.
8. Gemini generates the response and can call tools such as `check_inventory`, `get_price`, `add_to_cart`, and `apply_promo`.
9. The hallucination guard validates menu mentions.
10. Gemini TTS synthesizes response audio, and the PCM is written to the circular reference buffer used by AEC.
11. The browser receives `agent_response`, plays audio, highlights the FSM node, and updates cart totals.

## Socket.IO Events

Client to server:

- `start_session`: reset and begin a new voice lane session.
- `send_text`: send a typed or simulated transcript through the same agent path.
- `audio_frame`: stream raw PCM-16 audio frames.
- `audio_stream_stopped`: force finalization of any buffered speech.
- `reset_session`: clear backend and UI session state.
- `run_speaker_analytics`: run offline diarization over captured utterances.
- `update_settings`: update NLMS, VAD, and echo simulation parameters.

Server to client:

- `agent_response`: response text, cart, total, FSM node, guardrail flag, and optional PCM audio.
- `agent_error`: recoverable backend/model/TTS error.
- `asr_status`: microphone, VAD, and transcription status messages.
- `customer_speech`: transcript plus optional live speaker label.
- `dsp_levels`: raw/reference/clean audio peak levels for the dashboard.
- `speaker_analytics_result`: offline speaker clustering results.
- `session_reset`: acknowledgement that server state was reset.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create local configuration:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least:

```text
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```

For CPU-only Whisper, use:

```text
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

For GPU Whisper on Windows, keep `WHISPER_DEVICE=cuda` and ensure CUDA/CuDNN DLLs are discoverable. The app also checks pip-installed NVIDIA package DLL directories and `CT2_CUDA_DLL_DIRS`.

## Run

Use the runner:

```powershell
python run.py
```

Then open:

```text
http://localhost:5000
```

The first startup or first transcription can take time while Whisper, Silero, or SpeechBrain models are downloaded and cached.

## Configuration

Important `.env` settings:

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Required API key for Gemini LLM and Gemini TTS. |
| `GEMINI_LLM_MODEL` | Gemini model used by LangGraph nodes. |
| `GEMINI_LLM_TIMEOUT_SECONDS` | Request timeout for Gemini LLM calls. |
| `GEMINI_TTS_MODEL` | Gemini TTS model. |
| `GEMINI_TTS_VOICE` | Gemini TTS prebuilt voice name. |
| `ENABLE_GEMINI_TTS` | Set to `0` to skip backend TTS and continue text-only. |
| `WHISPER_MODEL_SIZE` | faster-whisper model size, such as `medium.en`. |
| `WHISPER_DEVICE` | `cuda` or `cpu`. |
| `WHISPER_COMPUTE_TYPE` | Typical values: `float16`, `int8`. |
| `WHISPER_CACHE_DIR` | Optional local model cache directory. |
| `USE_SILERO_VAD` | Set to `0` to use lightweight RMS energy VAD. |
| `PRELOAD_SILERO_VAD` | Load Silero at startup when using `run.py`. |
| `PRELOAD_WHISPER_ASR` | Load Whisper at startup when using `run.py`. |
| `LIVE_DIARIZATION` | Enable live speaker clustering for completed utterances. |
| `SPEAKER_ENROLLMENT_PATH` | Optional JSON file of enrolled speaker embeddings. |
| `SOCKETIO_ASYNC_MODE` | Defaults to `threading`; `eventlet` is opt-in. |
| `SERVER_HOST` / `SERVER_PORT` | Server bind host and port. |
| `DEBUG` | Flask/Socket.IO debug mode. |

## Menu and Tools

The menu registry lives in `voice_agent/config.py`. It includes item price, stock, and synonyms. The LLM can call tools in `voice_agent/llm/tools.py`:

- `check_inventory(item_id)`
- `get_price(item)`
- `add_to_cart(item, qty)`
- `apply_promo(code)`

Supported promo codes:

- `DISCOUNT10`: applies 10% off.
- `FREEFRIES`: adds free fries when available.

## Testing

Run the unit tests:

```powershell
python -m unittest discover -s tests
```

The current tests cover:

- Gemini API key environment resolution.
- VAD padding and empty-frame behavior.
- LangGraph node cart/total updates with mocked Gemini agent responses.

## Development Notes

- `run.py` is the preferred local entry point because it loads `.env`, logs runtime configuration, and can preload ASR/VAD.
- `app.py` also has a direct `__main__` block, but the runner is cleaner for development.
- The text simulator in the dashboard sends `send_text` events and bypasses ASR, which is useful for testing order logic without microphone/model latency.
- The microphone path and text simulator converge at `process_user_text`, so both exercise the same LangGraph agent flow after transcription.
- The hallucination guard runs before TTS so off-menu generated responses are corrected before being spoken.
- Speaker analytics are lazy-loaded because SpeechBrain can be heavy on first use.
=======
## Challenges and Solutions

During integration, a few runtime issues made the app appear to ignore code or
environment changes. The fixes below document what changed.

- Gemini model aliases caused confusing quota errors. The app now keeps model
  selection explicit through `GEMINI_LLM_MODEL` and `GEMINI_TTS_MODEL` in
  `.env` and `.env.example`.
- Gemini authentication failed with `401 UNAUTHENTICATED` when an OAuth-style
  credential was used as an API key. The app now reads only `GEMINI_KEY_API`
  and validates that it looks like a Google AI Studio API key before calling
  Gemini.
- Stale shell variables could override `.env`. Both `run.py` and `app.py` now
  call `load_dotenv(override=True)` so local `.env` values are the source of
  truth during development.
- Flask's debug reloader spawned extra processes, which made old code appear to
  be cached. `run.py` now starts Flask-SocketIO with `use_reloader=False`.
- A runtime debug endpoint was added to confirm the active backend process,
  app path, cwd, key prefix, and Gemini model config:

```text
http://127.0.0.1:5000/debug/runtime
```

- If code or `.env` changes do not appear to apply, stale listeners on port
  `5000` can be found and stopped with:

```powershell
netstat -ano | findstr :5000
Stop-Process -Id <PID>
```

- Gemini `429 RESOURCE_EXHAUSTED` errors are quota errors for the active Google
  project/key. The LLM path intentionally does not use a fallback.
- Gemini TTS failures are handled separately: they are logged and return a
  text-only response so the UI can continue.
- Gemini TTS now uses the streaming `google-genai` flow for
  `gemini-3.1-flash-tts-preview`, collects inline audio chunks, decodes the
  returned PCM MIME metadata, and resamples audio to the app's 16 kHz playback
  and echo-cancellation format.
- Browser voice turns were unreliable because microphone capture sample rates
  vary by device/browser, and Socket.IO binary payload shapes were inconsistent.
  The frontend now resamples mic input to 16 kHz PCM frames before ASR, while
  the backend returns TTS as base64 PCM with sample-rate metadata so audio
  responses play reliably in the browser.
- Audio model loading happened too late in the request path. Whisper could
  start downloading `medium.en` only after the first utterance was buffered,
  and Silero VAD could fail mid-stream on Torch Hub's interactive trust prompt.
  The app now preloads both models at startup, uses the cached Whisper snapshot
  when complete, downloads missing Whisper files only once, and loads Silero
  with an explicit trusted repository setting.
- Speaker analytics now imports SpeechBrain/PyTorch lazily, so normal server
  startup does not import `torch`.
>>>>>>> 49a559f33b1bf4eec9200e5476c09d90e7e007bc
