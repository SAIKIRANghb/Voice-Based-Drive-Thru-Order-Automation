# Drive-Thru Voice Agent

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
