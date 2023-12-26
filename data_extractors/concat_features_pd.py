import pandas as pd
import numpy as np


def get_mfcc(mfcc):
    mfcc = pd.read_csv(mfcc, sep=',')
    mfcc = mfcc.fillna(0)
    return mfcc

def get_librosa(librosa):
    librosa = pd.read_csv(librosa, sep=',')
    librosa = librosa.fillna(0)
    return librosa

def main(csv_mfcc, csv_librosa, csv_output):
    mfcc = get_mfcc(csv_mfcc)
    print(mfcc)

    print()

    librosa = get_librosa(csv_librosa)
    print(librosa)