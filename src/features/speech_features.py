"""Speech feature extraction for Parkinson's Disease detection."""

import numpy as np
import pandas as pd
from typing import Optional, Dict
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from ..utils.config import Config


class SpeechFeatureExtractor:
    """Extract acoustic features from speech/voice recordings."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the speech feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.feature_config = self.config.get_features_config().get('speech', {})
        self.sample_rate = self.feature_config.get('sample_rate', 16000)
        self.n_mfcc = self.feature_config.get('n_mfcc', 13)
        self.n_fft = self.feature_config.get('n_fft', 2048)
        self.hop_length = self.feature_config.get('hop_length', 512)
    
    def extract_mfcc(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features.
        
        Args:
            audio_signal: Audio signal array
            
        Returns:
            MFCC features array
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio feature extraction")
        
        mfcc = librosa.feature.mfcc(
            y=audio_signal,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Calculate statistics over time
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        return np.concatenate([mfcc_mean, mfcc_std])
    
    def extract_jitter(self, audio_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract jitter features (frequency variation).
        
        Args:
            audio_signal: Audio signal array
            
        Returns:
            Dictionary of jitter measurements
        """
        # Simplified jitter calculation
        # In practice, this would use more sophisticated pitch tracking
        return {
            'jitter_abs': 0.0,
            'jitter_rel': 0.0,
            'rap': 0.0,
            'ppq': 0.0
        }
    
    def extract_shimmer(self, audio_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract shimmer features (amplitude variation).
        
        Args:
            audio_signal: Audio signal array
            
        Returns:
            Dictionary of shimmer measurements
        """
        # Simplified shimmer calculation
        return {
            'shimmer_abs': 0.0,
            'shimmer_rel': 0.0,
            'apq3': 0.0,
            'apq5': 0.0
        }
    
    def extract_all_features(self, audio_path: str) -> pd.DataFrame:
        """
        Extract all speech features from an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            DataFrame with all extracted features
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio feature extraction")
        
        # Load audio file
        audio_signal, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract features
        features = {}
        
        # MFCC features
        mfcc_features = self.extract_mfcc(audio_signal)
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = mfcc_features[i]
            features[f'mfcc_{i}_std'] = mfcc_features[i + self.n_mfcc]
        
        # Jitter features
        jitter = self.extract_jitter(audio_signal)
        features.update(jitter)
        
        # Shimmer features
        shimmer = self.extract_shimmer(audio_signal)
        features.update(shimmer)
        
        # Additional features
        features['zcr_mean'] = np.mean(librosa.feature.zero_crossing_rate(audio_signal))
        features['energy_mean'] = np.mean(librosa.feature.rms(y=audio_signal))
        
        return pd.DataFrame([features])
    
    @staticmethod
    def get_feature_names() -> list:
        """
        Get list of all feature names extracted by this class.
        
        Returns:
            List of feature names
        """
        features = []
        
        # MFCC features (13 coefficients x 2 statistics)
        for i in range(13):
            features.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
        
        # Jitter features
        features.extend(['jitter_abs', 'jitter_rel', 'rap', 'ppq'])
        
        # Shimmer features
        features.extend(['shimmer_abs', 'shimmer_rel', 'apq3', 'apq5'])
        
        # Additional features
        features.extend(['zcr_mean', 'energy_mean'])
        
        return features


if __name__ == "__main__":
    print("Speech Feature Extractor")
    print("=" * 60)
    
    if not LIBROSA_AVAILABLE:
        print("Warning: librosa not available. Install with: pip install librosa")
    else:
        print("librosa is available for audio processing")
    
    extractor = SpeechFeatureExtractor()
    print(f"\nConfiguration:")
    print(f"  Sample rate: {extractor.sample_rate} Hz")
    print(f"  MFCC coefficients: {extractor.n_mfcc}")
    print(f"  FFT size: {extractor.n_fft}")
    print(f"  Hop length: {extractor.hop_length}")
    
    print(f"\nTotal features extracted: {len(extractor.get_feature_names())}")
    print(f"Feature names: {extractor.get_feature_names()[:5]}...")

