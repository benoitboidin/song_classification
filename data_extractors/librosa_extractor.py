"""
https://librosa.org/doc/main/feature.html#
"""


import librosa
import os
import numpy as np


def track_features(MUSIC_TRAIN_DIR, filename):
    waveform, sample_rate = librosa.load(MUSIC_TRAIN_DIR + filename)

    feature_tempo = librosa.feature.tempo(y=waveform, sr=sample_rate)
    feature_chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate, n_chroma=12) # ?
    feature_spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate) # ? 
    feature_spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sample_rate) # ?
    feature_zcr = librosa.feature.zero_crossing_rate(y=waveform) # ?

    features = {
        'tempo': feature_tempo,
        'chroma': np.asarray(feature_chroma).mean(axis=1),
        'spectral_centroid': np.asarray(feature_spectral_centroid).mean(axis=1),    
        'spectral_contrast': np.asarray(feature_spectral_contrast).mean(axis=1),
        'zcr': np.asarray(feature_zcr).mean(axis=1),
    }

    return features

def features_to_csv(MUSIC_DIR, output_filename):
    with open(output_filename, 'w') as f:
        
        header_tempo = ['tempo_{}'.format(i+1) for i in range(1)]
        header_chroma = ['chroma_{}'.format(i+1) for i in range(12)]
        header_spectral_centroid = ['spectral_centroid_{}'.format(i+1) for i in range(1)]
        header_spectral_contrast = ['spectral_contrast_{}'.format(i+1) for i in range(7)]
        header_zcr = ['zcr_{}'.format(i+1) for i in range(1)]
        header = ['track_id'] + header_tempo + header_chroma + header_spectral_centroid + header_spectral_contrast + header_zcr
        
        f.write(str(header))
        f.write('\n')

        for track_file in os.listdir(MUSIC_DIR):
            print('Extracting features from file: ', track_file)
            features = track_features(MUSIC_DIR, track_file)

            row = [track_file.replace('.mp3', '')] + features['tempo'].tolist() + features['chroma'].tolist() + features['spectral_centroid'].tolist() + features['spectral_contrast'].tolist() + features['zcr'].tolist()
            f.write(str(row))
            f.write('\n')


if __name__ == '__main__':

    # Test librosa features on a single track.
    # features = track_features('data/train.nosync/Train/', '000002.mp3')
    # print('\nChroma')
    # print(features['chroma'])
    # print('\nSpectral centroid')
    # print(features['spectral_centroid'])
    # print('\nSpectral contrast')
    # print(features['spectral_contrast'])
    # print('\nTempo')
    # print(features['tempo'])
    # print('\nZCR')
    # print(features['zcr'])

    features_to_csv('data/train.nosync/Train/', 'data/train_librosa_features.csv')