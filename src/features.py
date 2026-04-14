import librosa
import numpy as np


def extract_mel(audio_path, n_mels=128, max_len=128):
    """Load audio and compute a fixed-size log-mel spectrogram, z-score normalized.

    Args:
        audio_path: Path to a WAV (or other format ``librosa`` can read).
        n_mels: Number of mel frequency bins.
        max_len: Number of time frames to keep; shorter clips are zero-padded on the right.

    Returns:
        2D ``numpy.ndarray`` of shape ``(n_mels, max_len)``, dtype float, in dB scale
        with zero mean and unit variance (approximately), after per-array normalization.
    """
    y, sr = librosa.load(audio_path, sr=22050)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)

    # FIX SIZE (VERY IMPORTANT)
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)))
    else:
        mel_db = mel_db[:, :max_len]

    mean = mel_db.mean()
    std = mel_db.std()
    mel_db = (mel_db - mean) / (std + 1e-8)

    return mel_db
