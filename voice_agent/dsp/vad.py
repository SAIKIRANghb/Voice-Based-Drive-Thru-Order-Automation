import numpy as np
import logging
import os
import threading
from typing import Any, Tuple, cast

from voice_agent.config import SAMPLE_RATE, VAD_THRESHOLD

class VADGate:
    _silero_lock = threading.Lock()
    _silero_model = None
    _silero_utils = None
    _silero_load_failed = False

    def __init__(self, threshold=VAD_THRESHOLD, sample_rate=SAMPLE_RATE):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model: Any | None = None
        self.utils: Any | None = None
        self.initialized = False
        self.silero_load_failed = False
        
        # Energy threshold parameters for explicitly selected lightweight VAD.
        self.energy_threshold = 0.005 # Normalized RMS threshold
        self.use_silero = os.getenv("USE_SILERO_VAD", "1").lower() not in {"0", "false", "no"}
        
        if not self.use_silero:
            logging.info("Silero VAD disabled by USE_SILERO_VAD=0. Using energy-envelope VAD.")
            return

    def _energy_is_speech(self, pcm_frame: np.ndarray) -> bool:
        """Return speech activity using a lightweight RMS threshold."""
        float_frame = pcm_frame.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(float_frame**2)) if len(float_frame) > 0 else 0
        return bool(rms >= self.energy_threshold)

    def _load_silero(self):
        """Load Silero lazily so websocket connection setup stays responsive."""
        with self._silero_lock:
            if self._silero_model is not None:
                self.model = self._silero_model
                self.utils = self._silero_utils
                self.initialized = True
                return
            if self._silero_load_failed:
                raise RuntimeError("Silero VAD failed to load previously.")

            try:
                import torch
                import warnings
                warnings.filterwarnings('ignore')
                
                logging.info("Attempting to load Silero VAD model...")
                model, utils = cast(
                    Tuple[Any, Any],
                    torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        trust_repo=True,
                        verbose=False,
                    ),
                )
                type(self)._silero_model = model
                type(self)._silero_utils = utils
                self.model = model
                self.utils = utils
                self.initialized = True
                logging.info("Silero VAD successfully loaded.")
            except Exception as e:
                type(self)._silero_load_failed = True
                self.silero_load_failed = True
                logging.exception("Could not load Silero VAD.")
                raise RuntimeError("Could not load required Silero VAD model.") from e

    def is_speech(self, pcm_frame: np.ndarray) -> bool:
        """
        Detects if speech is present in a PCM frame.
        """
        if pcm_frame.size == 0:
            return False

        if self.use_silero:
            if not self.initialized:
                self._load_silero()
            if self.model is None:
                raise RuntimeError("Silero VAD model is not initialized.")
            import torch

            min_samples = int(np.ceil(self.sample_rate / 31.25))
            if pcm_frame.size < min_samples:
                pcm_frame = np.pad(pcm_frame, (0, min_samples - pcm_frame.size))

            # Normalize frame to [-1.0, 1.0] for Silero
            tensor = torch.FloatTensor(pcm_frame.astype(np.float32) / 32768.0)
            # Handle batch/channels (1D tensor expected by Silero).
            prob = self.model(tensor, self.sample_rate).item()
            return prob >= self.threshold
        
        # Energy-envelope VAD when explicitly selected.
        return self._energy_is_speech(pcm_frame)

    def gate(self, pcm_frame: np.ndarray) -> np.ndarray:
        """
        Zero-pads the frame if speech is not detected, preventing hallucination during silence.
        """
        if self.is_speech(pcm_frame):
            return pcm_frame
        else:
            return np.zeros_like(pcm_frame)
