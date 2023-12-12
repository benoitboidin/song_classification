"""
https://librosa.org/doc/main/feature.html#
"""


import librosa
import os


def track_features(MUSIC_TRAIN_DIR, filename):
    # Load the audio as a waveform `y` and sampling rate as `sr`
    y, sr = librosa.load(MUSIC_TRAIN_DIR + filename)

    feature_chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feature_spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feature_spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feature_rythm = librosa.feature.tempogram(y=y, sr=sr)
    feature_zcr = librosa.feature.zero_crossing_rate(y=y)
    feature_temporal = librosa.feature.tempogram(y=y)

    return feature_chroma, feature_spectral_centroid, feature_spectral_contrast, feature_rythm, feature_zcr, feature_temporal

def features_to_csv(MUSIC_DIR, output_filename):
    with open(output_filename, 'w') as f:
        for track_file in os.listdir(MUSIC_DIR):
            print('Extracting features from file: ', track_file)
            features = track_features(MUSIC_DIR, track_file)
            # track_file.replace('.mp3', ''), features
            row = track_file.replace('.mp3', ''), features
            f.write(str(row))
            f.write('\n')


if __name__ == '__main__':
    print(track_features('data/train.nosync/Train/', '000002.mp3'))
