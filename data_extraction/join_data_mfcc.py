"""
Create a new dataset with the track id, genre id, and mfcc features.
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
    train = train.fillna(0)
    test = test.fillna(0)
    return train, test

def main(train_filename, test_filename, train_mfcc_filename, test_mfcc_filename, MFCC_TRAIN_DIR, MFCC_TEST_DIR):
    print("\nReading data...")
    train, test = read_data(train_filename, test_filename)
    print("Reading MFCC data...")
    train_mfcc, test_mfcc = read_mfcc_data(MFCC_TRAIN_DIR, train, MFCC_TEST_DIR, test)
    print("Writing concatenated MFCC data...")
    train_mfcc.to_csv(train_mfcc_filename, index=False)
    test_mfcc.to_csv(test_mfcc_filename, index=False)
