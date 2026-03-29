import librosa
import numpy as np

def extract_mel(audio_path, n_mels=128, max_len=128):
    y, sr = librosa.load(audio_path, sr=22050)
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)
    
    # FIX SIZE (VERY IMPORTANT)
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0,0),(0,pad_width)))
    else:
        mel_db = mel_db[:, :max_len]
    
    return mel_db
