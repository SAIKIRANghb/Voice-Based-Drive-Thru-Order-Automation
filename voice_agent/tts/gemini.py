import os
import io
import re
import numpy as np
import logging
import soundfile as sf
import librosa
from voice_agent.tts.base import BaseTTS
from voice_agent.config import SAMPLE_RATE, get_gemini_api_key

class GeminiTTS(BaseTTS):
    DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"
    DEFAULT_VOICE = "Zephyr"

    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key or get_gemini_api_key()
        self.model_name = model_name or os.getenv("GEMINI_TTS_MODEL", self.DEFAULT_MODEL)
        self.client = None
        self.model = None
        
        if self.api_key:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)
            logging.info("Gemini TTS configured with google-genai model '%s'", self.model_name)
        else:
            logging.warning("No GEMINI_API_KEY set. GeminiTTS cannot synthesize audio.")

    @staticmethod
    def _normalize_for_speech(text: str) -> str:
        """Normalize UI-friendly order text into TTS-friendly spoken text."""
        def money(match):
            dollars = int(match.group(1))
            cents = int(match.group(2) or "0")
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            if cents == 0:
                return f"{dollars} {dollar_unit}"
            cent_unit = "cent" if cents == 1 else "cents"
            return f"{dollars} {dollar_unit} and {cents} {cent_unit}"

        normalized = re.sub(r"\$(\d+)(?:\.(\d{1,2}))?", money, text)
        normalized = normalized.replace(":", ".")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _parse_audio_mime_type(mime_type: str | None) -> dict[str, int]:
        bits_per_sample = 16
        rate = 24000

        for part in (mime_type or "").split(";"):
            param = part.strip()
            if param.lower().startswith("rate="):
                try:
                    rate = int(param.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif param.lower().startswith("audio/l"):
                try:
                    match = re.search(r"audio/L(\d+)", param, flags=re.IGNORECASE)
                    if match:
                        bits_per_sample = int(match.group(1))
                except (ValueError, IndexError):
                    pass

        return {"bits_per_sample": bits_per_sample, "rate": rate}

    @classmethod
    def _decode_audio(cls, audio_bytes: bytes, mime_type: str | None = None, sample_rate: int = 24000) -> np.ndarray:
        parsed_mime = cls._parse_audio_mime_type(mime_type)
        if mime_type:
            sample_rate = parsed_mime["rate"]

        try:
            pcm, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        except Exception:
            if parsed_mime["bits_per_sample"] != 16:
                raise ValueError(f"Unsupported Gemini TTS PCM bit depth: {parsed_mime['bits_per_sample']}")
            pcm = np.frombuffer(audio_bytes, dtype=np.int16)
            sr = sample_rate

        if len(pcm.shape) > 1:
            pcm = np.mean(pcm.astype(np.float32), axis=1).astype(np.int16)

        if sr != SAMPLE_RATE:
            pcm = librosa.resample(
                pcm.astype(np.float32),
                orig_sr=sr,
                target_sr=SAMPLE_RATE,
            )
            pcm = np.clip(pcm, -32768, 32767).astype(np.int16)

        return pcm.astype(np.int16)

    @staticmethod
    def _extract_audio_bytes(response) -> tuple[bytes, str | None]:
        candidates = getattr(response, "candidates", None) or []
        finish_reasons = []
        for candidate in candidates:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                finish_reasons.append(str(finish_reason))
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            for part in parts or []:
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None) if inline_data is not None else None
                if data:
                    return data, getattr(inline_data, "mime_type", None)

        details = f" Finish reasons: {', '.join(finish_reasons)}." if finish_reasons else ""
        raise ValueError(f"No audio content returned by Gemini TTS.{details}")

    @staticmethod
    def _iter_response_parts(response):
        parts = getattr(response, "parts", None)
        if parts is not None:
            yield from parts
            return

        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            yield from (getattr(content, "parts", None) or [])

    def _stream_audio_bytes(self, speech_text: str) -> tuple[bytes, str | None]:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=float(os.getenv("GEMINI_TTS_TEMPERATURE", "1")),
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=os.getenv("GEMINI_TTS_VOICE", self.DEFAULT_VOICE),
                    )
                )
            ),
        )
        chunks = []
        mime_type = None
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"## Transcript:\n{speech_text}")],
                )
            ],
            config=config,
        ):
            for part in self._iter_response_parts(chunk):
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None) if inline_data is not None else None
                if data:
                    chunks.append(data)
                    mime_type = mime_type or getattr(inline_data, "mime_type", None)

        if not chunks:
            raise ValueError("No audio content returned by Gemini TTS stream.")
        return b"".join(chunks), mime_type

    def synthesize(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot synthesize empty text.")
        speech_text = self._normalize_for_speech(text)
            
        try:
            if self.client is not None:
                audio_bytes, mime_type = self._stream_audio_bytes(speech_text)
                return self._decode_audio(audio_bytes, mime_type=mime_type)

            raise RuntimeError("Gemini TTS client is not initialized.")
            
        except Exception as e:
            logging.error(f"Gemini TTS synthesis failed: {e}")
            raise
