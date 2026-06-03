import unittest

import numpy as np

from voice_agent.dsp.vad import VADGate


class FakeSileroModel:
    def __init__(self):
        self.last_size = None

    def __call__(self, tensor, sample_rate):
        self.last_size = tensor.numel()
        return tensor.new_tensor(0.7)


class VADGateTests(unittest.TestCase):
    def test_silero_input_is_padded_to_minimum_supported_size(self):
        model = FakeSileroModel()
        vad = VADGate(threshold=0.6, sample_rate=16000)
        vad.initialized = True
        vad.model = model

        self.assertTrue(vad.is_speech(np.ones(320, dtype=np.int16)))
        self.assertEqual(model.last_size, 512)

    def test_empty_frame_is_not_speech(self):
        vad = VADGate(threshold=0.6, sample_rate=16000)

        self.assertFalse(vad.is_speech(np.array([], dtype=np.int16)))


if __name__ == "__main__":
    unittest.main()
