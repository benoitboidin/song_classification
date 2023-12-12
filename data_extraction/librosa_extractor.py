"""
https://librosa.org/doc/main/feature.html#
"""


import librosa
import os
import numpy as np


def track_features(MUSIC_TRAIN_DIR, filename):
    # Load the audio as a waveform `y` and sampling rate as `sr`
    y, sr = librosa.load(MUSIC_TRAIN_DIR + filename)

    feature_tempo = librosa.feature.tempo(y=y, sr=sr)

    feature_chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12) # ?
    feature_spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr) # ? 
    feature_spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr) # ?
    feature_zcr = librosa.feature.zero_crossing_rate(y=y) # ?
    feature_temporal = librosa.feature.tempogram(y=y) # ?

    features = {
        'tempo': feature_tempo,

        'chroma': np.asarray(feature_chroma).mean(axis=1),
        'spectral_centroid': np.asarray(feature_spectral_centroid).mean(axis=1),    
        'spectral_contrast': np.asarray(feature_spectral_contrast).mean(axis=1),
        'zcr': np.asarray(feature_zcr).mean(axis=1),
        'temporal': np.asarray(feature_temporal).mean(axis=1),
    }

    return features

def features_to_csv(MUSIC_DIR, output_filename):
    with open(output_filename, 'w') as f:
        for track_file in os.listdir(MUSIC_DIR):
            print('Extracting features from file: ', track_file)
            features = track_features(MUSIC_DIR, track_file)
            # features = [feature.tolist() for feature in features]
            # track_file.replace('.mp3', ''), features
            row = track_file.replace('.mp3', ''), features
            f.write(str(row))
            f.write('\n')


if __name__ == '__main__':
    features = track_features('data/train.nosync/Train/', '000002.mp3')

    print('\nChroma')
    print(features['chroma'])
    print('\nSpectral centroid')
    print(features['spectral_centroid'])
    print('\nSpectral contrast')
    print(features['spectral_contrast'])
    print('\nTempo')
    print(features['tempo'])
    print('\nZCR')
    print(features['zcr'])
    print('\nTemporal')
    print(features['temporal'])
