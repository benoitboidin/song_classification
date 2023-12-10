"""
Create a new dataset with the track id, genre id, and mfcc features.

TRAINING DATA
track if and genre id are in train.csv
mfcc features are in mfcc/train/

TESTING DATA
track id is in test.csv
mfcc features are in mfcc/test/
"""

import os
import pandas as pd


def read_data(train_filename, test_filename):
    
    train = pd.read_csv(train_filename)
    test = pd.read_csv(test_filename)
    
    return train, test

def read_mfcc_data(mfcc_train_dir, train, mfcc_test_dir, test):

    # Read every file in the mfcc directory
    train_mfcc = []
    test_mfcc = []

    for filename in os.listdir(mfcc_train_dir):
        with open(mfcc_train_dir + filename, 'r') as f:
            mfcc = f.read()
            mfcc = mfcc.replace('\n', '').split(' ')[:-1]
            mfcc = [float(x) for x in mfcc]
            train_mfcc.append(mfcc)

    for filename in os.listdir(mfcc_test_dir):
        with open(mfcc_test_dir + filename, 'r') as f:
            mfcc = f.read()
            mfcc = mfcc.replace('\n', '').split(' ')[:-1]
            mfcc = [float(x) for x in mfcc]
            test_mfcc.append(mfcc)

    # Concatenate the mfcc features with the track id and genre id
    train = pd.concat([train, pd.DataFrame(train_mfcc)], axis=1)
    test = pd.concat([test, pd.DataFrame(test_mfcc)], axis=1)
    
    return train, test


if __name__ == '__main__':
    train_filename = 'train.csv'
    test_filename = 'test.csv'
    output_filename = 'train_mfcc.csv'

    MFCC_TRAIN_DIR = 'mfcc/train/'
    MFCC_TEST_DIR = 'mfcc/test/'

    print("Reading data...")
    train, test = read_data(train_filename, test_filename)

    print("Reading mfcc data...")
    train_mfcc, test_mfcc = read_mfcc_data(MFCC_TRAIN_DIR, train, MFCC_TEST_DIR, test)

    print(train_mfcc)
    print(test_mfcc)