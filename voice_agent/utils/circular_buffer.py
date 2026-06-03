import numpy as np
import threading
from collections import deque
from voice_agent.config import REFERENCE_BUFFER_SAMPLES, SAMPLE_RATE

class CircularReferenceBuffer:
    def __init__(self, maxlen=REFERENCE_BUFFER_SAMPLES):
        """
        Thread-safe circular buffer for storing reference audio samples.
        Default maxlen = 2 seconds at 16kHz.
        """
        self.maxlen = maxlen
        self.lock = threading.Lock()
        # Initialize deque with zeros so we always have references even at start
        self.buffer = deque([0] * maxlen, maxlen=maxlen)

    def write(self, data: np.ndarray):
        """Write new reference audio frames to the buffer."""
        with self.lock:
            # Handle float arrays or int16
            samples = data.tolist() if hasattr(data, 'tolist') else list(data)
            self.buffer.extend(samples)

    def get_latest(self, num_samples: int) -> np.ndarray:
        """
        Retrieve the latest N samples. If the buffer has fewer samples than requested,
        pads with zeros at the beginning.
        """
        with self.lock:
            # Slice the end of the deque
            buffer_len = len(self.buffer)
            if num_samples <= buffer_len:
                # Get the last num_samples
                start_idx = buffer_len - num_samples
                slice_list = [self.buffer[i] for i in range(start_idx, buffer_len)]
            else:
                # Pad with zeros
                pad_len = num_samples - buffer_len
                slice_list = [0] * pad_len + list(self.buffer)
                
            return np.array(slice_list, dtype=np.int16)

    def duration_seconds(self) -> float:
        """Return the configured look-back duration for diagnostics."""
        return self.maxlen / SAMPLE_RATE

    def get_reference_slice(self, taps: int, frame_size: int) -> np.ndarray:
        """
        Get the reference slice of length (taps + frame_size) for NLMS updates.
        This provides the exact history window needed to match the incoming mic frame.
        """
        total_needed = taps + frame_size
        return self.get_latest(total_needed)

    def clear(self):
        """Reset the buffer with zeros."""
        with self.lock:
            self.buffer.clear()
            self.buffer.extend([0] * self.maxlen)
