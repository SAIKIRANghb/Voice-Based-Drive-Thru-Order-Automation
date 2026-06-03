import os
import base64
import logging
import faulthandler
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv(override=True)

CRASH_LOG_PATH = os.getenv("CRASH_LOG_PATH", "crash.log")
try:
    _crash_log = open(CRASH_LOG_PATH, "a", buffering=1)
    faulthandler.enable(file=_crash_log, all_threads=True)
except Exception as e:
    logging.warning("Could not enable faulthandler crash log at %s: %s", CRASH_LOG_PATH, e)

SOCKETIO_ASYNC_MODE = os.getenv("SOCKETIO_ASYNC_MODE", "threading")
if SOCKETIO_ASYNC_MODE == "eventlet":
    import eventlet

    # Eventlet is opt-in because monkey patching can destabilize SDK/native clients.
    eventlet.monkey_patch()

from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np

from voice_agent.dsp.aec import NLMSFilter
from voice_agent.dsp.vad import VADGate
from voice_agent.asr.whisper import WhisperASR
from voice_agent.tts.gemini import GeminiTTS
from voice_agent.llm.graph import run_agent_turn
from voice_agent.utils.circular_buffer import CircularReferenceBuffer
from voice_agent.config import (
    NLMS_EPS,
    NLMS_MU,
    NLMS_TAPS,
    REFERENCE_BUFFER_SAMPLES,
    SAMPLE_RATE,
    VAD_THRESHOLD,
    FSM_STATE_TIMEOUT_MS,
    get_gemini_api_key,
)

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "drive_thru_secret_key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=SOCKETIO_ASYNC_MODE)

# Session store: keeps tracking components per websocket client connection ID (sid)
sessions = {}

# Initialize real model components lazily so server startup remains responsive.
whisper_asr = None
speaker_verifier = None
speaker_diarizer = None
gemini_tts = None

def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}

def get_tts_engine():
    global gemini_tts
    if gemini_tts is None:
        gemini_tts = GeminiTTS()
    return gemini_tts

def get_speaker_verifier():
    global speaker_verifier
    if speaker_verifier is None:
        from voice_agent.spkrec.embedding import SpeakerVerifier

        enrollment_path = os.getenv("SPEAKER_ENROLLMENT_PATH")
        speaker_verifier = SpeakerVerifier(enrollment_path=enrollment_path)
    return speaker_verifier

def get_speaker_diarizer():
    global speaker_diarizer
    if speaker_diarizer is None:
        from voice_agent.spkrec.diarization import SpeakerDiarizer

        verifier = get_speaker_verifier()
        speaker_diarizer = SpeakerDiarizer(verifier)
    return speaker_diarizer

def get_session_components(sid):
    if sid not in sessions:
        sessions[sid] = {
            # LLM LangGraph conversation state
            "state": {
                "messages": [],
                "cart": {},
                "total_price": 0.0,
                "current_node": "greeting",
                "next_node": "greeting",
                "discount": 0.0,
                "free_items": [],
                "last_response": "",
                "hallucination_warning": False,
                "state_timeout_ms": FSM_STATE_TIMEOUT_MS["greeting"],
            },
            # DSP Echo Cancellation
            "aec": NLMSFilter(taps=NLMS_TAPS, mu=NLMS_MU, eps=NLMS_EPS),
            "vad": VADGate(threshold=VAD_THRESHOLD, sample_rate=SAMPLE_RATE),
            "circ_buf": CircularReferenceBuffer(maxlen=REFERENCE_BUFFER_SAMPLES),
            "analytics_utterances": [],
            "speaker_profiles": [],
            
            # Sound stream states
            "mic_buffer": [], # Accumulates clean samples during speech
            "is_speech_active": False,
            "silence_counter": 0,
            "silence_samples": 0,
            "session_active": False,
            "audio_inactive_warned": False,
            "audio_error_warned": False,
            "audio_frame_count": 0,
            "speech_frame_count": 0,
            "vad_idle_notice_count": 0,
            
            # Settings
            "inject_echo": True,
            "echo_delay": int(SAMPLE_RATE * 0.01), # ~10ms at 16kHz
            "echo_scale": 0.4
        }
    return sessions[sid]

def make_initial_state():
    return {
        "messages": [],
        "cart": {},
        "total_price": 0.0,
        "current_node": "greeting",
        "next_node": "greeting",
        "discount": 0.0,
        "free_items": [],
        "last_response": "",
        "hallucination_warning": False,
        "state_timeout_ms": FSM_STATE_TIMEOUT_MS["greeting"],
    }

def response_payload(updated_state, audio_bytes: bytes):
    return {
        "text": updated_state["last_response"],
        "cart": updated_state["cart"],
        "total": updated_state["total_price"],
        "node": updated_state["current_node"],
        "timeout_ms": updated_state["state_timeout_ms"],
        "hallucination_warning": updated_state["hallucination_warning"],
        "audio": base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else "",
        "audio_encoding": "pcm_s16le",
        "audio_sample_rate": SAMPLE_RATE,
    }

def reset_session(sess):
    sess["state"] = make_initial_state()
    sess["circ_buf"].clear()
    sess["aec"].reset()
    sess["mic_buffer"] = []
    sess["is_speech_active"] = False
    sess["silence_counter"] = 0
    sess["silence_samples"] = 0
    sess["audio_inactive_warned"] = False
    sess["audio_error_warned"] = False
    sess["audio_frame_count"] = 0
    sess["speech_frame_count"] = 0
    sess["vad_idle_notice_count"] = 0
    sess["analytics_utterances"] = []
    sess["speaker_profiles"] = []

def start_session_payload(sid: str):
    sess = get_session_components(sid)
    reset_session(sess)
    sess["session_active"] = True
    logging.info("Running initial LangGraph turn for %s.", sid)
    updated_state = run_agent_turn("", sess["state"])
    logging.info("Initial LangGraph turn completed for %s: node=%s, text=%r", sid, updated_state.get("current_node"), updated_state.get("last_response"))
    sess["state"] = updated_state
    logging.info("Synthesizing initial TTS for %s.", sid)
    audio_bytes = synthesize_and_buffer_or_empty(updated_state["last_response"], sess, sid)
    if audio_bytes:
        logging.info("Initial TTS completed for %s: %s bytes.", sid, len(audio_bytes))
    return response_payload(updated_state, audio_bytes)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/debug/runtime')
def debug_runtime():
    try:
        gemini_key = get_gemini_api_key()
        gemini_key_prefix = gemini_key[:6]
        if os.getenv("GEMINI_API_KEY"):
            gemini_key_source = "GEMINI_API_KEY"
        elif os.getenv("GEMINI_KEY_API"):
            gemini_key_source = "GEMINI_KEY_API"
        else:
            gemini_key_source = "unknown"
    except Exception as e:
        gemini_key_prefix = ""
        gemini_key_source = f"invalid: {e}"

    return {
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "app_file": __file__,
        "gemini_llm_model": os.getenv("GEMINI_LLM_MODEL"),
        "gemini_tts_model": os.getenv("GEMINI_TTS_MODEL"),
        "gemini_key_prefix": gemini_key_prefix,
        "gemini_key_source": gemini_key_source,
    }

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.after_request
def add_dev_cache_headers(response):
    if os.getenv("FLASK_ENV") == "development":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    logging.info(f"Client connected: {sid}")
    get_session_components(sid)

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    logging.info(f"Client disconnected: {sid}")
    if sid in sessions:
        del sessions[sid]

@socketio.on('start_session')
def handle_start_session(data=None):
    sid = request.sid
    logging.info("Session started for %s with real LLM/TTS/ASR components.", sid)
    
    try:
        payload = start_session_payload(sid)
    except Exception as e:
        logging.exception("Failed to start LangGraph session for %s", sid)
        emit('agent_error', {"message": str(e)})
        return
        
    emit('agent_response', payload)

@socketio.on('send_text')
def handle_send_text(data):
    """Handles text input directly from the text simulator panel."""
    sid = request.sid
    data = data or {}
    try:
        response = process_user_text(sid, data.get("text", "").strip())
        if response is not None:
            emit('agent_response', response)
    except Exception as e:
        logging.exception("Text turn failed for %s", sid)
        emit('agent_error', {"message": str(e)})

@socketio.on('reset_session')
def handle_reset_session():
    sid = request.sid
    sess = get_session_components(sid)
    reset_session(sess)
    sess["session_active"] = False
    logging.info("Session reset for %s", sid)
    emit('session_reset')


def process_user_text(sid: str, user_text: str):
    """Runs one customer text turn through LangGraph and TTS."""
    sess = get_session_components(sid)
    if not sess.get("session_active", False):
        reset_session(sess)
        sess["session_active"] = True
    if not user_text:
        return None
    
    logging.info(f"Received text from client {sid}: {user_text}")
    
    try:
        updated_state = run_agent_turn(user_text, sess["state"])
    except Exception as e:
        logging.exception("LangGraph turn failed for %s", sid)
        emit('agent_error', {"message": str(e)})
        return None

    sess["state"] = updated_state
    
    response_text = updated_state["last_response"]
    audio_bytes = synthesize_and_buffer_or_empty(response_text, sess, sid)

    return response_payload(updated_state, audio_bytes)

def finalize_buffered_speech(sid: str, sess, reason: str = "silence"):
    if not sess["mic_buffer"]:
        emit('asr_status', {"message": "No buffered speech to transcribe."})
        return

    sess["is_speech_active"] = False
    sess["silence_counter"] = 0
    sess["silence_samples"] = 0

    full_phrase = np.array(sess["mic_buffer"], dtype=np.int16)
    sess["mic_buffer"] = []
    if full_phrase.size > 0:
        sess["analytics_utterances"].append(full_phrase.copy())
    speaker_info = diarize_live_utterance(full_phrase, sess)

    emit('asr_status', {"message": f"Speech {reason}. Transcribing {len(full_phrase) / SAMPLE_RATE:.1f}s of audio..."})
    transcription = trigger_transcription(full_phrase, sess)
    if transcription.strip():
        emit('customer_speech', {"text": transcription, **speaker_info})
        response = process_user_text(sid, transcription)
        if response is not None:
            emit('agent_response', response)
    else:
        emit('asr_status', {"message": "No speech text detected. Try speaking a little louder or closer to the mic."})

@socketio.on('audio_frame')
def handle_audio_frame(data):
    """
    Handles raw PCM-16 (16kHz mono) chunks from browser mic.
    Applies echo simulation, NLMS filter, and VAD check.
    """
    sid = request.sid
    sess = get_session_components(sid)
    if data is None:
        return

    if not sess.get("session_active", False):
        if not sess.get("audio_inactive_warned", False):
            sess["audio_inactive_warned"] = True
            emit('asr_status', {"message": "Audio is arriving, but the voice session is not active yet. Start the lane session first."})
        return

    try:
        # Decode audio payload (raw binary)
        raw_pcm = np.frombuffer(data, dtype=np.int16)
        if raw_pcm.size == 0:
            return
        sess["audio_frame_count"] += 1
        
        # 1. Echo Injection Simulation (if toggled)
        # Grab the reference signal from our pre-DAC buffer
        ref_buffer = sess["circ_buf"].get_latest(sess["aec"].taps + len(raw_pcm))
        
        if sess["inject_echo"]:
            # Simulate acoustic loopback echo
            mic_input = sess["aec"].simulate_echo(
                mic_clean=raw_pcm, 
                reference_pcm=ref_buffer,
                delay_samples=sess["echo_delay"],
                echo_scale=sess["echo_scale"]
            )
        else:
            mic_input = raw_pcm
            
        # 2. Run NLMS Acoustic Echo Cancellation
        # Get reference slice
        ref_slice = sess["circ_buf"].get_reference_slice(sess["aec"].taps, len(raw_pcm))
        clean_audio = sess["aec"].process_frame(mic_input, ref_slice)
        
        # Calculate energy levels for visualization dashboard
        raw_level = int(np.max(np.abs(mic_input.astype(np.int32)))) if len(mic_input) > 0 else 0
        ref_tail = ref_slice[-len(raw_pcm):] if len(raw_pcm) > 0 else np.array([], dtype=np.int16)
        ref_level = int(np.max(np.abs(ref_tail.astype(np.int32)))) if len(ref_tail) > 0 else 0
        clean_level = int(np.max(np.abs(clean_audio.astype(np.int32)))) if len(clean_audio) > 0 else 0
        
        # Emit levels for the dashboard graph
        emit('dsp_levels', {
            "raw": raw_level,
            "ref": ref_level,
            "clean": clean_level
        })
        if sess["audio_frame_count"] == 1:
            emit('asr_status', {"message": f"Audio frames are reaching the server ({len(raw_pcm)} samples/frame)."})
        
        # 3. Voice Activity Detection (VAD) Gate
        is_speech = sess["vad"].is_speech(clean_audio)
        gated_audio = clean_audio if is_speech else np.zeros_like(clean_audio)
        
        if is_speech:
            if not sess["is_speech_active"]:
                emit('asr_status', {"message": "Speech detected. Listening until you pause..."})
            sess["is_speech_active"] = True
            sess["speech_frame_count"] += 1
            sess["silence_counter"] = 0
            sess["silence_samples"] = 0
            sess["mic_buffer"].extend(gated_audio.tolist())
        else:
            if sess["is_speech_active"]:
                sess["silence_counter"] += 1
                sess["silence_samples"] += len(gated_audio)
                sess["mic_buffer"].extend(gated_audio.tolist())
                
                # About 0.9s of silence signifies end-of-speech.
                if sess["silence_samples"] >= int(SAMPLE_RATE * 0.9):
                    finalize_buffered_speech(sid, sess, "ended")
            else:
                if (
                    sess["audio_frame_count"] % 150 == 0
                    and sess["vad_idle_notice_count"] < 3
                ):
                    sess["vad_idle_notice_count"] += 1
                    emit('asr_status', {
                        "message": f"Audio is being processed, but VAD has not detected speech yet. Raw peak={raw_level}, clean peak={clean_level}, threshold={sess['vad'].threshold}."
                    })
    except Exception as e:
        logging.exception("Audio frame processing failed for %s", sid)
        if not sess.get("audio_error_warned", False):
            sess["audio_error_warned"] = True
            emit('asr_status', {"message": f"Audio input was received but could not be processed: {e}"})

@socketio.on('audio_stream_stopped')
def handle_audio_stream_stopped(data=None):
    sid = request.sid
    sess = get_session_components(sid)
    if not sess.get("session_active", False):
        return
    if sess.get("is_speech_active") or sess.get("mic_buffer"):
        try:
            finalize_buffered_speech(sid, sess, "stopped")
        except Exception as e:
            logging.exception("Audio stream stop finalization failed for %s", sid)
            emit('asr_status', {"message": f"Audio input was received but could not be transcribed: {e}"})
    else:
        emit('asr_status', {"message": "Microphone stopped. No speech was buffered for transcription."})

def trigger_transcription(audio_data, sess) -> str:
    """Executes real Whisper ASR on accumulated speech data."""
    # Use real Whisper ASR for transcription
    global whisper_asr
    if whisper_asr is None:
        emit('asr_status', {"message": "Loading Whisper ASR model. First transcription can take a while if the model is not cached."})
        whisper_asr = WhisperASR()
        emit('asr_status', {"message": "Whisper ASR model loaded. Decoding speech..."})
    else:
        emit('asr_status', {"message": "Decoding speech with Whisper ASR..."})
    return whisper_asr.transcribe(audio_data)

def preload_whisper_asr():
    """Load ASR at startup so model cache/download work happens before mic use."""
    global whisper_asr
    if whisper_asr is None:
        logging.info("Preloading Whisper ASR model...")
        whisper_asr = WhisperASR()
        logging.info("Whisper ASR preload complete.")
    return whisper_asr

def preload_silero_vad():
    """Load the required Silero VAD at startup instead of first audio frame."""
    if not env_flag("USE_SILERO_VAD", "1"):
        logging.info("Silero VAD preload skipped because USE_SILERO_VAD=0.")
        return None

    logging.info("Preloading Silero VAD model...")
    vad = VADGate(threshold=VAD_THRESHOLD, sample_rate=SAMPLE_RATE)
    vad._load_silero()
    logging.info("Silero VAD preload complete.")
    return vad

def diarize_live_utterance(audio_data: np.ndarray, sess) -> dict:
    """Assign a live speaker cluster to one completed utterance."""
    if not env_flag("LIVE_DIARIZATION", "1"):
        return {"speaker": "Customer", "speaker_cluster": None, "speaker_score": 0.0}
    min_samples = int(SAMPLE_RATE * float(os.getenv("LIVE_DIARIZATION_MIN_SECONDS", "0.7")))
    speech_samples = audio_data[np.nonzero(audio_data)]
    if speech_samples.size < min_samples:
        return {"speaker": "Speaker ?", "speaker_cluster": None, "speaker_score": 0.0}

    try:
        verifier = get_speaker_verifier()
        embedding = verifier.get_embedding(audio_data)
    except Exception as e:
        logging.exception("Live diarization failed; continuing without speaker label: %s", e)
        return {"speaker": "Speaker ?", "speaker_cluster": None, "speaker_score": 0.0}

    profiles = sess.setdefault("speaker_profiles", [])
    match_threshold = float(os.getenv("LIVE_DIARIZATION_SIMILARITY", "0.45"))
    best_index = None
    best_score = -1.0

    for idx, profile in enumerate(profiles):
        score = float(np.dot(embedding, profile["centroid"]))
        if score > best_score:
            best_score = score
            best_index = idx

    if best_index is None or best_score < match_threshold:
        cluster_id = len(profiles)
        profiles.append({"centroid": embedding, "count": 1})
        best_score = 1.0
    else:
        cluster_id = best_index
        profile = profiles[cluster_id]
        count = profile["count"]
        centroid = ((profile["centroid"] * count) + embedding) / (count + 1)
        norm = np.linalg.norm(centroid)
        profile["centroid"] = centroid / norm if norm > 0 else centroid
        profile["count"] = count + 1

    speaker_name = "unknown"
    enrolled_score = 0.0
    for name, ref_vec in verifier.enrollment.items():
        score = float(np.dot(embedding, ref_vec))
        if score > enrolled_score:
            speaker_name = name
            enrolled_score = score

    speaker_label = f"Speaker {cluster_id + 1}"
    if speaker_name != "unknown":
        speaker_label = f"{speaker_label} ({speaker_name})"

    logging.info("Live diarization: %s similarity=%.3f enrolled=%s %.3f", speaker_label, best_score, speaker_name, enrolled_score)
    return {
        "speaker": speaker_label,
        "speaker_cluster": cluster_id,
        "speaker_score": max(best_score, enrolled_score),
    }

def synthesize_and_buffer(response_text, sess) -> bytes:
    """Synthesizes TTS and writes PCM to the AEC reference buffer before playback."""
    if not env_flag("ENABLE_GEMINI_TTS", "1"):
        raise RuntimeError("ENABLE_GEMINI_TTS=0 disables required voice synthesis.")

    tts_audio = get_tts_engine().synthesize(response_text)
    sess["circ_buf"].write(tts_audio)
    return tts_audio.tobytes()

def synthesize_and_buffer_or_empty(response_text, sess, sid: str) -> bytes:
    """Best-effort TTS: keep the conversation moving when the voice API fails."""
    try:
        return synthesize_and_buffer(response_text, sess)
    except Exception as e:
        logging.exception("TTS synthesis failed for %s; continuing without audio.", sid)
        emit('agent_error', {"message": f"TTS unavailable: {e}"})
        return b""

def identify_speaker(audio_data):
    """Runs SpeechBrain ECAPA speaker identification for offline analytics."""
    verifier = get_speaker_verifier()
    speaker_name, speaker_score = verifier.identify(audio_data)
    logging.info("Speaker identification: %s (%.3f)", speaker_name, speaker_score)
    return speaker_name, speaker_score

@socketio.on('run_speaker_analytics')
def handle_run_speaker_analytics():
    """Runs ECAPA speaker identification and offline diarization clustering."""
    sid = request.sid
    sess = get_session_components(sid)
    utterances = sess.get("analytics_utterances", [])
    
    if not utterances:
        emit('speaker_analytics_result', {"results": []})
        return
        
    try:
        diarizer = get_speaker_diarizer()
        # Diarize/cluster the accumulated customer utterances
        diarize_results = diarizer.diarize_utterances(utterances, distance_threshold=0.08)
        
        results = []
        for item in diarize_results:
            idx = item["utterance_index"]
            utt = utterances[idx]
            lbl = f"Speaker {item['cluster_id']}"
            if item["speaker_name"] != "unknown":
                lbl += f" ({item['speaker_name']})"
            results.append({
                "utterance_index": idx,
                "speaker": lbl,
                "speaker_score": item["speaker_score"],
                "duration_ms": int(len(utt) / SAMPLE_RATE * 1000),
            })
            
        emit('speaker_analytics_result', {"results": results})
    except Exception as e:
        logging.error("Failed to run speaker diarization analytics: %s", e)
        emit('agent_error', {"message": f"Diarization failed: {str(e)}"})

@socketio.on('update_settings')
def handle_update_settings(data):
    sid = request.sid
    sess = get_session_components(sid)
    data = data or {}
    
    # Update NLMS adaptive filter parameters
    sess["aec"].update_params(
        taps=int(data.get("taps", NLMS_TAPS)),
        mu=float(data.get("mu", NLMS_MU)),
        eps=float(data.get("eps", NLMS_EPS))
    )
    
    # Update VAD threshold
    sess["vad"].threshold = float(data.get("vad_threshold", VAD_THRESHOLD))
    
    # Update Echo injection
    sess["inject_echo"] = bool(data.get("inject_echo", sess["inject_echo"]))
    sess["echo_delay"] = int(data.get("echo_delay", int(SAMPLE_RATE * 0.01)))
    sess["echo_scale"] = float(data.get("echo_scale", 0.4))
    
    logging.info(f"Updated settings for connection {sid}: {sess['aec'].taps} taps, mu={sess['aec'].mu}, VAD gate={sess['vad'].threshold}, Echo={sess['inject_echo']}")

if __name__ == '__main__':
    # Run the Eventlet WSGI server
    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "5000"))
    debug = env_flag("DEBUG", "1")
    logging.info("Starting Drive-Thru Voice Agent server on http://%s:%s", server_host, server_port)
    logging.info(
        "Runtime config: pid=%s model=%s cwd=%s",
        os.getpid(),
        os.getenv("GEMINI_LLM_MODEL"),
        os.getcwd(),
    )
    socketio.run(app, host=server_host, port=server_port, debug=debug, use_reloader=False, allow_unsafe_werkzeug=True)
