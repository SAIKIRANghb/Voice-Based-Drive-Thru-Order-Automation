import os
import unittest
from unittest.mock import patch

from voice_agent.tts.gemini import GeminiTTS


class GeminiTTSTests(unittest.TestCase):
    def test_normalizes_old_pro_tts_alias(self):
        env = {
            "GEMINI_API_KEY": "",
            "GEMINI_TTS_MODEL": "gemini-2.5-pro-tts",
        }
        with patch.dict(os.environ, env, clear=True):
            tts = GeminiTTS(api_key="test-key")

        self.assertEqual(tts.model_name, "gemini-2.5-pro-preview-tts")

    def test_normalizes_old_flash_tts_alias(self):
        env = {
            "GEMINI_API_KEY": "",
            "GEMINI_TTS_MODEL": "gemini-2.5-flash-tts",
        }
        with patch.dict(os.environ, env, clear=True):
            tts = GeminiTTS(api_key="test-key")

        self.assertEqual(tts.model_name, "gemini-2.5-flash-preview-tts")


if __name__ == "__main__":
    unittest.main()
