import numpy as np

class BaseTTS:
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize the input text into a PCM-16 (16kHz, mono) audio waveform.
        Returns:
            np.ndarray: int16 array of audio samples.
        """
        raise NotImplementedError("TTS Strategy subclasses must implement synthesize()")
