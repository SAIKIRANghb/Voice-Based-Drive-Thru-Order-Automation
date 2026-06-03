import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from voice_agent.spkrec.embedding import SpeakerVerifier
from voice_agent.config import SAMPLE_RATE

class SpeakerDiarizer:
    """
    Offline/Analytics Speaker Diarization module.
    Utilizes a SpeakerVerifier (SpeechBrain ECAPA-TDNN) to compute 192-dim
    embeddings and groups them via Scipy Agglomerative Hierarchical Clustering.
    """
    def __init__(self, verifier: SpeakerVerifier):
        self.verifier = verifier

    def diarize_utterances(
        self, 
        utterances: List[np.ndarray], 
        distance_threshold: float = 0.08
    ) -> List[Dict[str, Any]]:
        """
        Diarize a list of separate audio utterances (gated customer speech segments).
        Extracts embeddings for each utterance, clusters them, and matches
        each segment to enrolled speaker identities.
        
        Args:
            utterances: List of 1D numpy arrays (PCM-16 mono @ 16kHz).
            distance_threshold: Linkage distance threshold for fcluster (1.0 - cosine similarity).
            
        Returns:
            A list of dictionaries, one for each utterance, containing:
                - utterance_index: int
                - cluster_id: int (speaker cluster group index, 0-based)
                - speaker_name: str (identified identity name or 'unknown')
                - speaker_score: float (cosine similarity score to identified enrolled speaker)
        """
        num_utterances = len(utterances)
        if num_utterances == 0:
            logging.info("Diarize: No utterances to analyze.")
            return []

        # 1. Extract speaker embeddings
        embeddings = []
        for idx, utt in enumerate(utterances):
            emb = self.verifier.get_embedding(utt)
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        # 2. Perform clustering using linkage
        if num_utterances == 1:
            cluster_labels = np.array([0])
        else:
            # Pairwise cosine distance: dist = 1 - cosine_similarity
            # Since embeddings are normalized, dist = 1 - dot(u, v)
            dists = pdist(embeddings, metric='cosine')

            # Perform average linkage hierarchical clustering
            Z = linkage(dists, method='average')

            # Flatten the hierarchy tree using the distance threshold
            cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance') - 1

        # 3. Build results mapping each utterance to a cluster and matching to enrolled speakers
        results = []
        for i in range(num_utterances):
            cluster_id = int(cluster_labels[i])
            utt_pcm = utterances[i]
            
            # Cross-reference with SpeakerVerifier to identify if they match any enrolled speaker
            speaker_name, speaker_score = self.verifier.identify(utt_pcm)
            
            results.append({
                "utterance_index": i,
                "cluster_id": cluster_id,
                "speaker_name": speaker_name,
                "speaker_score": float(speaker_score),
            })
            
        logging.info(f"Diarized {num_utterances} utterances into {len(set(cluster_labels))} unique speaker clusters.")
        return results

    def segment_audio(self, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> List[Tuple[int, int]]:
        """
        Helper to segment continuous PCM audio into active speech intervals (start_sample, end_sample).
        Uses a simple energy envelope with hangover frames to group speech blocks.
        """
        # Calculate RMS energy in 20ms frames
        frame_size = int(sample_rate * 0.02) # 320 samples
        if frame_size == 0 or len(pcm) < frame_size:
            return []
            
        num_frames = len(pcm) // frame_size
        energy_threshold = 0.005 # RMS energy threshold for active speech
        
        speech_frames = []
        for f in range(num_frames):
            frame = pcm[f * frame_size : (f + 1) * frame_size]
            float_frame = frame.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(float_frame**2)) if len(float_frame) > 0 else 0
            speech_frames.append(rms >= energy_threshold)

        # Group contiguous active speech frames
        segments = []
        in_speech = False
        start_idx = 0
        hangover = 15 # 300ms hangover to prevent clipping words
        silence_count = 0
        
        for idx, is_active in enumerate(speech_frames):
            if is_active:
                if not in_speech:
                    in_speech = True
                    start_idx = idx * frame_size
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= hangover:
                        in_speech = False
                        end_idx = (idx - hangover + 1) * frame_size
                        # Only keep segments longer than 0.3s
                        if end_idx - start_idx >= int(sample_rate * 0.3):
                            segments.append((start_idx, end_idx))
                            
        if in_speech:
            segments.append((start_idx, num_frames * frame_size))
            
        return segments

    def diarize_continuous_audio(
        self, 
        pcm: np.ndarray, 
        sample_rate: int = SAMPLE_RATE,
        distance_threshold: float = 0.08
    ) -> List[Dict[str, Any]]:
        """
        Diarize a single continuous recording of PCM audio.
        Segments the audio via VAD/energy, extracts embeddings for each segment,
        clusters them, and returns timestamps and speaker ids.
        
        Returns a list of dicts, one for each segmented speech interval:
            - start_ms: start time of segment in milliseconds
            - end_ms: end time of segment in milliseconds
            - cluster_id: int (clustered speaker ID, 0-based)
            - speaker_name: str (identified enrolled speaker or 'unknown')
            - speaker_score: float (identification score)
        """
        if len(pcm) == 0:
            return []
            
        # 1. Segment continuous audio
        sample_segments = self.segment_audio(pcm, sample_rate)
        if not sample_segments:
            logging.info("Continuous Diarize: No active speech segments detected.")
            return []
            
        # 2. Extract segment sub-arrays
        utterances = [pcm[start:end] for start, end in sample_segments]
        
        # 3. Cluster and identify speaker identities using the diarize_utterances pipeline
        cluster_results = self.diarize_utterances(utterances, distance_threshold)
        
        # 4. format results with timestamps
        diarized_segments = []
        for idx, item in enumerate(cluster_results):
            start_sample, end_sample = sample_segments[idx]
            diarized_segments.append({
                "segment_index": idx,
                "start_ms": int((start_sample / sample_rate) * 1000),
                "end_ms": int((end_sample / sample_rate) * 1000),
                "cluster_id": item["cluster_id"],
                "speaker_name": item["speaker_name"],
                "speaker_score": item["speaker_score"],
            })
            
        return diarized_segments
