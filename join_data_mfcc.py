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


def read_data(train_filename, test_filename):
    
    with open(train_filename, 'r') as f:
        train = f.read().split('\n')
        train = [_.split(',') for _ in train]
        train = train[:-1]
    
    with open(test_filename, 'r') as f:
        test = f.read().split('\n')
        test = [_.split(',') for _ in test]
        test = test[:-1]
    
    return train, test

def read_mfcc_data(mfcc_train_dir, train, mfcc_test_dir, test):

    # Read every file in the mfcc directory
    for i_row, file in enumerate(os.listdir(mfcc_train_dir)):
        with open(mfcc_train_dir + file, 'r') as f:
            mfcc = f.read().replace('\n', '').split(' ')[:-1]
            mfcc = [float(_) for _ in mfcc]
            train[i_row].extend(mfcc)
    
    for i_row, file in enumerate(os.listdir(mfcc_test_dir)):
        with open(mfcc_test_dir + file, 'r') as f:
            mfcc = f.read().replace('\n', '').split(' ')[:-1]
            mfcc = [float(_) for _ in mfcc]
            test[i_row].extend(mfcc)
    
    return train, test


if __name__ == '__main__':
    train_filename = 'train.csv'
    test_filename = 'test.csv'
    output_filename = 'train_mfcc.csv'

    MFCC_TRAIN_DIR = 'mfcc/train/'
    MFCC_TEST_DIR = 'mfcc/test/'

    print("Reading data...")
    train, test = read_data(train_filename, test_filename)
    train, test = read_mfcc_data(MFCC_TRAIN_DIR, train, MFCC_TEST_DIR, test)

    for row in train:
        print(row)
