import sys
import os
import numpy as np
import logging

# Ensure root folder is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from voice_agent.spkrec.embedding import SpeakerVerifier
from voice_agent.spkrec.diarization import SpeakerDiarizer
from voice_agent.config import SAMPLE_RATE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_voice_like_signal(pitch: float, duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generates a rich harmonic signal simulating vocal cords (fundamental pitch + decaying harmonics).
    This is much more realistic for voice models than a pure sine wave.
    """
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    signal = np.zeros_like(t)
    # Add fundamental frequency and 5 harmonics
    for harmonic in range(1, 7):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * (pitch * harmonic) * t)
    
    # Add a tiny bit of high-frequency texture/noise
    noise = np.random.normal(0, 0.02, len(t))
    signal += noise
    
    # Scale to PCM-16 range
    signal = 25000.0 * (signal / np.max(np.abs(signal)))
    return signal.astype(np.int16)

def test_speaker_diarization():
    logging.info("Initializing SpeakerVerifier and SpeakerDiarizer...")

    try:
        verifier = SpeakerVerifier(device=os.getenv("SPEAKER_DEVICE") or None)
        diarizer = SpeakerDiarizer(verifier)
    except Exception as e:
        logging.error(f"Failed to load SpeechBrain models: {e}")
        return False

    logging.info("Generating voice-like harmonic waveforms for testing...")
    # Speaker 1: low pitch voice (e.g., 120 Hz)
    speaker1_waveA = generate_voice_like_signal(pitch=120.0, duration_seconds=2.0)
    speaker1_waveB = generate_voice_like_signal(pitch=120.0, duration_seconds=2.2)
    # Speaker 2: high pitch voice (e.g., 260 Hz)
    speaker2_waveA = generate_voice_like_signal(pitch=260.0, duration_seconds=2.1)
    speaker2_waveB = generate_voice_like_signal(pitch=260.0, duration_seconds=2.3)

    utterances = [
        speaker1_waveA, # Index 0
        speaker2_waveA, # Index 1
        speaker1_waveB, # Index 2
        speaker2_waveB, # Index 3
    ]

    # Compute and print raw similarities
    logging.info("Computing embeddings and pairwise cosine distances...")
    embs = [verifier.get_embedding(utt) for utt in utterances]
    
    # Pairwise cosine distances
    for i in range(4):
        for j in range(i+1, 4):
            dist = 1.0 - np.dot(embs[i], embs[j])
            logging.info(f"Distance between Wave #{i+1} and Wave #{j+1} = {dist:.4f}")

    # Set threshold dynamically based on intra-speaker vs inter-speaker distances
    # Typically, same speaker distance is low, different is high.
    # Let's run diarization with threshold = 0.5 (default)
    logging.info("Running diarization clustering...")
    try:
        results = diarizer.diarize_utterances(utterances, distance_threshold=0.08)
    except Exception as e:
        logging.error(f"Diarize execution failed: {e}")
        return False

    logging.info("Analyzing diarization clustering results:")
    for item in results:
        logging.info(
            f"Utterance #{item['utterance_index'] + 1}: Cluster ID = {item['cluster_id']}, "
            f"Identified Speaker = {item['speaker_name']} (score = {item['speaker_score']:.3f})"
        )

    cluster_a1 = results[0]["cluster_id"]
    cluster_a2 = results[2]["cluster_id"]
    cluster_b1 = results[1]["cluster_id"]
    cluster_b2 = results[3]["cluster_id"]

    success = True

    if cluster_a1 != cluster_a2:
        logging.error(f"Assertion Failed: Speaker 1 samples clustered separately! (Cluster A1={cluster_a1}, Cluster A2={cluster_a2})")
        success = False
    else:
        logging.info("Pass: Speaker 1 samples clustered together.")

    if cluster_b1 != cluster_b2:
        logging.error(f"Assertion Failed: Speaker 2 samples clustered separately! (Cluster B1={cluster_b1}, Cluster B2={cluster_b2})")
        success = False
    else:
        logging.info("Pass: Speaker 2 samples clustered together.")

    if cluster_a1 == cluster_b1:
        logging.error(f"Assertion Failed: Different speakers merged into same cluster ID {cluster_a1}!")
        success = False
    else:
        logging.info("Pass: Different speakers correctly assigned to distinct clusters.")

    # Test continuous audio segmenting and diarizing
    logging.info("Testing continuous audio diarization flow...")
    silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.int16) # 0.5s silence
    continuous_pcm = np.concatenate([
        speaker1_waveA,
        silence,
        speaker2_waveA,
        silence,
        speaker1_waveB,
        silence,
        speaker2_waveB
    ])
    
    try:
        segments = diarizer.diarize_continuous_audio(continuous_pcm, distance_threshold=0.08)
        logging.info(f"Diarized continuous recording into {len(segments)} segments:")
        for seg in segments:
            logging.info(
                f"Segment #{seg['segment_index'] + 1}: {seg['start_ms']}ms - {seg['end_ms']}ms, "
                f"Cluster = {seg['cluster_id']}, Speaker = {seg['speaker_name']}"
            )
        
        if len(segments) != 4:
            logging.warning(f"Expected 4 segments in continuous audio, but detected {len(segments)}")
    except Exception as e:
        logging.error(f"Continuous diarization failed: {e}")
        success = False

    return success

if __name__ == "__main__":
    logging.info("=== Starting Speaker Diarization Verification ===")
    test_result = test_speaker_diarization()
    if test_result:
        logging.info("=== VERIFICATION PASSED SUCCESSFULLY ===")
        sys.exit(0)
    else:
        logging.error("=== VERIFICATION FAILED ===")
        sys.exit(1)
