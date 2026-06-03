import numpy as np
from voice_agent.config import NLMS_EPS, NLMS_MU, NLMS_TAPS

class NLMSFilter:
    def __init__(self, taps=NLMS_TAPS, mu=NLMS_MU, eps=NLMS_EPS):
        self.taps = taps
        self.mu = mu
        self.eps = eps
        # Weights are initialized to zeros
        self.w = np.zeros(self.taps, dtype=np.float64)
        
    def reset(self):
        """Reset the adaptive filter weights to zero."""
        self.w = np.zeros(self.taps, dtype=np.float64)
        
    def update_params(self, taps=None, mu=None, eps=None):
        """Update filter parameters at runtime."""
        if taps is not None and taps != self.taps:
            self.taps = taps
            self.w = np.zeros(self.taps, dtype=np.float64)
        if mu is not None:
            self.mu = mu
        if eps is not None:
            self.eps = eps

    def process_frame(self, d: np.ndarray, x_buf: np.ndarray) -> np.ndarray:
        """
        Process a 20ms frame of microphone audio.
        d: float64 or int16 array of microphone samples (desired signal: speaker speech + echo)
        x_buf: reference signal samples of length taps + len(d).
        Returns the echo-cancelled output frame (int16).
        """
        L = len(d)
        if len(x_buf) < self.taps + L:
            # Pad reference buffer if it's too short
            padding = (self.taps + L) - len(x_buf)
            x_buf = np.pad(x_buf, (padding, 0), 'constant')
        
        # Ensure we work with float64 internally for numerical stability
        d_float = d.astype(np.float64)
        x_float = x_buf.astype(np.float64)
        
        out = np.zeros(L, dtype=np.float64)
        
        for i in range(L):
            # Extract the N past samples of the reference signal for the i-th sample
            x_n = x_float[i : i + self.taps]
            
            # Predict the echo
            # w is taps-long, x_n is taps-long. Estimate = w @ x_n
            y_n = np.dot(self.w, x_n)
            
            # Error (clean signal estimate)
            e_n = d_float[i] - y_n
            
            # Update filter weights: w(n+1) = w(n) + [mu * e(n) * x(n)] / [eps + ||x(n)||^2]
            norm_x = np.dot(x_n, x_n)
            self.w += (self.mu * e_n * x_n) / (self.eps + norm_x)
            
            out[i] = e_n
            
        # Clip to prevent overflow before converting back to int16
        out = np.clip(out, -32768, 32767)
        return out.astype(np.int16)

    def simulate_echo(self, mic_clean: np.ndarray, reference_pcm: np.ndarray, delay_samples=160, echo_scale=0.4) -> np.ndarray:
        """
        Utility to simulate physical echo leakage into the microphone.
        Combines clean mic signal (e.g. user talking) with delayed and attenuated reference signal.
        """
        L = len(mic_clean)
        # Create an echo signal from the reference
        echo = np.zeros(L, dtype=np.float64)
        
        # Pull reference signal with delay
        if len(reference_pcm) >= delay_samples + L:
            # Extract reference with delay
            ref_slice = reference_pcm[-delay_samples - L : -delay_samples]
            if len(ref_slice) == L:
                echo = ref_slice.astype(np.float64) * echo_scale
        
        # Mix clean mic signal and echo
        mixed = mic_clean.astype(np.float64) + echo
        mixed = np.clip(mixed, -32768, 32767)
        return mixed.astype(np.int16)
