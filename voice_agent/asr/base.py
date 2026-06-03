import numpy as np

class BaseASR:
    def transcribe(self, pcm_data: np.ndarray) -> str:
        """
        Transcribe the input PCM-16 (16kHz, mono) audio waveform to text.
        """
        raise NotImplementedError("ASR Strategy subclasses must implement transcribe()")
