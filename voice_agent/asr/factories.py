from voice_agent.asr.base import BaseASR


def build_asr_engine() -> BaseASR:
    from voice_agent.asr.whisper import WhisperASR

    return WhisperASR()
