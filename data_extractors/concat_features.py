"""
Create a new dataset with the track id, genre id, and mfcc features.
"""


import os
import pandas as pd


def read_mfcc_files(mfcc_train_dir, train, mfcc_test_dir, test):
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

    # Add label to the mfcc features instead of 1, 2, 3...
    train_mfcc = pd.DataFrame(train_mfcc).sort_values(by=[0])
    test_mfcc = pd.DataFrame(test_mfcc).sort_values(by=[0])

    columns={i: 'mfcc_{}'.format(i+1) for i in range(len(train_mfcc.columns))}
    train_mfcc = train_mfcc.rename(columns=columns)
    test_mfcc = test_mfcc.rename(columns=columns)


    # Concatenate the mfcc features with the track id and genre id
    train = pd.concat([train, train_mfcc], axis=1)
    test = pd.concat([test, test_mfcc], axis=1)
    train = train.fillna(0)
    test = test.fillna(0)
    return train, test

def read_mfcc_features(mfcc_train, train, mfcc_test, test):
    
    columns=['mfcc_{}'.format(i+1) for i in range(23)]

    train_mfcc = pd.read_csv(mfcc_train, sep=' ', names=columns).drop(columns=['mfcc_23'])
    test_mfcc = pd.read_csv(mfcc_test, sep=' ', names=columns).drop(columns=['mfcc_23'])
    
    train = pd.concat([train, train_mfcc], axis=1)
    test = pd.concat([test, test_mfcc], axis=1)
    train = train.fillna(0)
    test = test.fillna(0)

    # TODO: Insert lines 113025, 155298, 155306 

    return train, test

def read_librosa_features(librosa_train, train, librosa_test, test):
    train_librosa = pd.read_csv(librosa_train)
    test_librosa = pd.read_csv(librosa_test)

    columns={i: 'librosa_{}'.format(i+1) for i in range(len(train_librosa.columns))}
    train_librosa = train_librosa.rename(columns=columns)
    test_librosa = test_librosa.rename(columns=columns)
    
    train = pd.concat([train, train_librosa], axis=1)
    test = pd.concat([test, test_librosa], axis=1)

    train = train.fillna(0)
    test = test.fillna(0)

    return train, test


def main(train_filename, test_filename, train_mfcc_filename, test_mfcc_filename):

    print("\nReading data...")
    train = pd.read_csv(train_filename)
    test = pd.read_csv(test_filename)

    train_mfcc, test_mfcc = read_mfcc_features(train_mfcc_filename, train, test_mfcc_filename, test)
    # TODO: Concat librosa features.

    print("Writing concatenated data...")
    print(train_mfcc)
    train_mfcc.to_csv('data/train_features.csv', index=False, header=True)
    test_mfcc.to_csv('data/test_features.csv', index=False, header=True)


if __name__ == '__main__':
    main('data/train.csv',
        'data/test.csv',
        'data/mfcc/train.csv',
        'data/mfcc/test.csv')