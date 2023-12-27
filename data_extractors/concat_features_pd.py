import pandas as pd
import numpy as np


def get_mfcc(mfcc):
    mfcc = pd.read_csv(mfcc, sep=',')
    mfcc = mfcc.fillna(0)
    return mfcc

def get_librosa(librosa):
    librosa = pd.read_csv(librosa, sep=',')
    librosa.drop(['track_id'], axis=1, inplace=True)
    librosa = librosa.fillna(0)
    return librosa

def main(csv_mfcc, csv_librosa, csv_output):
    mfcc = get_mfcc(csv_mfcc)
    print(mfcc)
    librosa = get_librosa(csv_librosa)
    print(librosa)
    concat = pd.concat([mfcc, librosa], axis=1)
    concat.to_csv(csv_output, index=False)


if __name__ == '__main__':
    main('data/train_mfcc.csv',
         'data/train_librosa.csv',
         'data/train_features.csv')
    
    main('data/test_mfcc.csv',
        'data/test_librosa.csv',
        'data/test_features.csv')
