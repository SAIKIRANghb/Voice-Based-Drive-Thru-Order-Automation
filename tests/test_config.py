import os
import unittest
from unittest.mock import patch

from voice_agent.config import get_gemini_api_key


class GeminiApiKeyConfigTests(unittest.TestCase):
    def test_prefers_gemini_api_key(self):
        env = {
            "GEMINI_API_KEY": "AIzaPreferred",
            "GEMINI_KEY_API": "AIzaLegacy",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_gemini_api_key(), "AIzaPreferred")

    def test_uses_legacy_key_when_preferred_is_missing(self):
        with patch.dict(os.environ, {"GEMINI_KEY_API": "AIzaLegacy"}, clear=True):
            self.assertEqual(get_gemini_api_key(), "AIzaLegacy")

    def test_uses_legacy_key_when_preferred_is_empty(self):
        env = {
            "GEMINI_API_KEY": "",
            "GEMINI_KEY_API": "AIzaLegacy",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_gemini_api_key(), "AIzaLegacy")

    def test_rejects_blank_preferred_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": " "}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY is required"):
                get_gemini_api_key()


if __name__ == "__main__":
    unittest.main()
