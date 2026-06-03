import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np


class SpeakerVerifier:
    """Loads SpeechBrain ECAPA-TDNN and computes speaker embeddings."""

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        enrollment_path: Optional[str] = None,
        sample_rate: int = 16000,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate

        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        import torch

        cache_root = os.getenv("SPEECHBRAIN_CACHE_DIR") or os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "speechbrain",
        )
        cache_dir = os.path.join(cache_root, "spkrec-ecapa-voxceleb")
        run_opts = {"device": device or ("cuda" if torch.cuda.is_available() else "cpu")}

        logging.info("Loading SpeechBrain speaker model '%s'...", self.model_name)
        self.classifier = EncoderClassifier.from_hparams(
            source=self.model_name,
            savedir=cache_dir,
            run_opts=run_opts,
            local_strategy=LocalStrategy.COPY
        )
        logging.info("SpeechBrain speaker model loaded on %s.", run_opts["device"])

        self.enrollment: Dict[str, np.ndarray] = {}
        if enrollment_path and os.path.isfile(enrollment_path):
            with open(enrollment_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for name, vec in raw.items():
                self.enrollment[name] = self._normalize(np.array(vec, dtype=np.float32))

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def get_embedding(self, pcm: np.ndarray) -> np.ndarray:
        """Returns a normalized embedding for 16 kHz mono PCM audio."""
        if pcm.size == 0:
            raise ValueError("Cannot compute a speaker embedding from empty audio.")

        if pcm.dtype == np.int16:
            waveform = pcm.astype(np.float32) / 32768.0
        else:
            waveform = pcm.astype(np.float32)

        import torch

        wav = torch.from_numpy(waveform).unsqueeze(0)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(wav)

        return self._normalize(embedding.squeeze().detach().cpu().numpy())

    def identify(self, pcm: np.ndarray) -> Tuple[str, float]:
        """Returns the best enrolled speaker and cosine similarity."""
        if not self.enrollment:
            return "unknown", 0.0

        query = self.get_embedding(pcm)
        best_name = "unknown"
        best_score = -1.0

        for name, ref_vec in self.enrollment.items():
            score = float(np.dot(query, ref_vec))
            if score > best_score:
                best_name = name
                best_score = score

        return best_name, best_score
