import numpy as np
import librosa

def extract_feature(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extracts a feature vector consistent with the training pipeline.
    
    Features extracted:
      - 40 MFCCs (mean over time)
      - Chroma features (mean over time; typically 12 values)
      - Mel-spectrogram features (mean over time; typically 128 values)
    
    Returns:
      A 1D numpy array combining the three sets of features (expected shape ~ (180,)).
    """
    # Ensure the audio is sampled at 44100 Hz
    TARGET_SAMPLE_RATE = 44100
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE

    # Extract 40 MFCCs and take the mean over time (result: 40 values)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Extract chroma features and take the mean over time (result: typically 12 values)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Extract mel-spectrogram and take the mean over time (result: typically 128 values)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)

    # Combine features into one vector
    feature_vector = np.hstack([mfccs_mean, chroma_mean, mel_mean])
    return feature_vector
