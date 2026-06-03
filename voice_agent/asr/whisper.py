import os
import sys
import numpy as np
import logging
from pathlib import Path
from voice_agent.asr.base import BaseASR
from voice_agent.config import SAMPLE_RATE

_CUDA_DLL_DIR_HANDLES = []


def _add_windows_cuda_dll_directories():
    """Make CUDA runtime DLLs from pip/system installs visible to CTranslate2."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_root = site_packages / "nvidia"
    candidates = []

    explicit_dirs = os.getenv("CT2_CUDA_DLL_DIRS", "")
    candidates.extend(Path(path) for path in explicit_dirs.split(os.pathsep) if path)

    cuda_path = os.getenv("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    candidates.extend(
        [
            nvidia_root / "cublas" / "bin",
            nvidia_root / "cuda_runtime" / "bin",
            nvidia_root / "cudnn" / "bin",
        ]
    )

    if nvidia_root.is_dir():
        candidates.extend(nvidia_root.glob("*/*/bin"))

    for dll_dir in candidates:
        if not dll_dir.is_dir():
            continue
        dll_dir_str = str(dll_dir)
        if dll_dir_str not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = dll_dir_str + os.pathsep + os.environ.get("PATH", "")
        try:
            _CUDA_DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir_str))
            logging.info("Added CUDA DLL directory for faster-whisper: %s", dll_dir_str)
        except OSError as e:
            logging.warning("Could not add CUDA DLL directory %s: %s", dll_dir_str, e)



def ensure_whisper_model_cached(model_size: str, cache_dir: str | None = None) -> str:
    """Return a local faster-whisper model path, downloading it only when absent."""
    from faster_whisper.utils import download_model

    model_path = download_model(
        model_size,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    logging.info("faster-whisper model '%s' loaded from %s.", model_size, model_path)
    return model_path

class WhisperASR(BaseASR):
    def __init__(self, model_size=None, device=None, compute_type=None):
        self.model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "medium.en")
        self.device = device or os.getenv("WHISPER_DEVICE") or self._default_device()
        self.compute_type = compute_type or os.getenv("WHISPER_COMPUTE_TYPE") or (
            "float16" if self.device == "cuda" else "int8"
        )
        self.model = None
        
        _add_windows_cuda_dll_directories()

        from faster_whisper import WhisperModel

        cache_dir = os.getenv("WHISPER_CACHE_DIR") or None
        model_path = ensure_whisper_model_cached(self.model_size, cache_dir=cache_dir)
        logging.info(
            "Loading faster-whisper model from '%s' on %s (%s)...",
            model_path,
            self.device,
            self.compute_type,
        )
        self.model = WhisperModel(
            model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        logging.info("faster-whisper model loaded successfully.")

    @staticmethod
    def _default_device() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def compute_log_mel(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Compute the LLD Whisper log-Mel representation: 80 bins, 400 FFT, 160 hop."""
        import librosa

        stft = librosa.stft(
            pcm.astype(np.float32) / 32768.0,
            n_fft=400,
            hop_length=160,
            win_length=400,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
        magnitude = np.abs(stft) ** 2
        mel_fb = librosa.filters.mel(
            sr=sr,
            n_fft=400,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0,
        )
        mel_spec = mel_fb @ magnitude
        return np.log10(np.maximum(mel_spec, 1e-10))

    def transcribe(self, pcm_data: np.ndarray) -> str:
        if self.model is None:
            raise RuntimeError("Whisper ASR model is not initialized.")

        # Convert int16 to float32 normalized
        float_data = pcm_data.astype(np.float32) / 32768.0

        # transcribe expects a numpy array or a path
        segments, info = self.model.transcribe(
            float_data,
            beam_size=1,            # greedy decoding for speed
            language="en",
            word_timestamps=True,
            vad_filter=False        # L2 already applies NLMS + VAD gating
        )

        return " ".join([segment.text for segment in segments]).strip()
