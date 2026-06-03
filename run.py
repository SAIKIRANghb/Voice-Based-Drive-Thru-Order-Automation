import os
import logging
from dotenv import load_dotenv

# Set logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load local environment variables (like GEMINI_API_KEY)
load_dotenv(override=True)

from app import app, env_flag, preload_silero_vad, preload_whisper_asr, socketio

if __name__ == '__main__':
    logging.info("Initializing Drive-Thru Voice Agent Runner...")
    logging.info(
        "Runtime config: pid=%s model=%s cwd=%s",
        os.getpid(),
        os.getenv("GEMINI_LLM_MODEL"),
        os.getcwd(),
    )
    if env_flag("PRELOAD_SILERO_VAD", "1"):
        preload_silero_vad()
    if env_flag("PRELOAD_WHISPER_ASR", "1"):
        preload_whisper_asr()
    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "5000"))
    debug = env_flag("DEBUG", "1")
    socketio.run(
        app,
        host=server_host,
        port=server_port,
        debug=debug,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
