"""
https://librosa.org/doc/main/feature.html#
"""


import librosa


def main(MUSIC_TRAIN_DIR):
    # Load the audio as a waveform `y` and sampling rate as `sr`
    y, sr = librosa.load(MUSIC_TRAIN_DIR + '000002.mp3')

    feature_chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feature_spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feature_spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feature_rythm = librosa.feature.tempogram(y=y, sr=sr)
    feature_zcr = librosa.feature.zero_crossing_rate(y=y)
    feature_temporal = librosa.feature.tempogram(y=y)

