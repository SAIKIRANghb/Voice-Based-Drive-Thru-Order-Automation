from voice_agent.tts.base import BaseTTS


def build_tts_engine() -> BaseTTS:
    from voice_agent.tts.gemini import GeminiTTS

    return GeminiTTS()
