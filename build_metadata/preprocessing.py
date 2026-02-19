import librosa

TARGET_SR = 22050
TRIM_DB = 20


def preprocess_audio(file_path):

    # Load
    y, sr = librosa.load(file_path, sr=None)

    # Resample
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # Convert to mono
    y = librosa.to_mono(y)

    # Remove silence
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)

    # Normalize
    y = librosa.util.normalize(y)

    return y
