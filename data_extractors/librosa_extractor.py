"""
Extract mp3 track features using librosa library.
https://librosa.org/doc/main/feature.html#
"""


import librosa
import os
import numpy as np
import pandas as pd
import threading
import csv


"""
Extract features from a track using librosa library.
"""
def get_track_features(filepath): 
    waveform, sample_rate = librosa.load(filepath)

    features_mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=20)
    feature_tempo = librosa.feature.tempo(y=waveform, sr=sample_rate)
    feature_chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate, n_chroma=12)
    feature_spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    feature_spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sample_rate)
    feature_zcr = librosa.feature.zero_crossing_rate(y=waveform)

    features = np.hstack((0, #tmp
                          np.asarray(features_mfcc).mean(axis=1), 
                          feature_tempo, 
                          np.asarray(feature_chroma).mean(axis=1), 
                          np.asarray(feature_spectral_centroid).mean(axis=1), 
                          np.asarray(feature_spectral_contrast).mean(axis=1),  
                          np.asarray(feature_zcr).mean(axis=1))).ravel()
    
    features = features.tolist()
    features[0] = filepath[-10:-4]
    return features


"""
Create header corresponding to the features extracted by get_track_features().
"""
def get_features_header():
    header = ['track_id']
    header.append(['mfcc_' + str(i) for i in range(1, 21)])
    header.append('tempo')
    header.extend(['chroma_' + str(i) for i in range(1, 13)])
    header.append('spectral_centroid')
    header.extend(['spectral_contrast_' + str(i) for i in range(1, 8)])
    header.append('zcr')

    header = np.hstack(header).ravel().tolist()

    return header


"""
Extract features from a list of tracks using librosa library.
"""
def get_tracks_features(tracks_dir, tracks_list):
    features = []
    test = 3
    for track in tracks_list:
        if test == 0:
            break
        test -= 1
        filepath = os.path.join(tracks_dir, track) + '.mp3'
        features.append(get_track_features(filepath))
    # print('Progress: {}/{}'.format(track, len([_ for _ in os.listdir(tracks_dir)])), flush=True, end='\r')
        print()
        print(track)
        print(features)

    # Sort features by track name.
    features_df = pd.DataFrame(np.array(features), columns=get_features_header())
    features_df = features_df.sort_values(by=['track_id'])
    return features_df




if __name__ == '__main__': 
    # Test
    tracks_dir = '/Users/benoitboidin/Desktop/s9_info/traitement_son_musique/project/data/test.nosync/Test'
    tracks_list = []
    with open('data/test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            tracks_list.append(row[0])
    features = get_tracks_features(tracks_dir, tracks_list)
    print(features)
    features.to_csv('data/test_features.csv', index=False)

    # Train
    tracks_dir = '/Users/benoitboidin/Desktop/s9_info/traitement_son_musique/project/data/train.nosync/Train'
    tracks_list = []
    with open('data/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            tracks_list.append(row[0])
    features = get_tracks_features(tracks_dir, tracks_list)
    print(features)
    features.to_csv('data/train_features.csv', index=False)
