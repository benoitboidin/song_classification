import os


if __name__ == '__main__':

    # Get data from files.
    mfcc_test = []
    filenames = sorted(os.listdir('data/mfcc/test'))
    for file in filenames:
        with open('data/mfcc/test/' + file, 'r') as f:
            mfcc_test.append(f.read())

    mfcc_train = []
    filenames = sorted(os.listdir('data/mfcc/train'))
    for file in filenames:
        with open('data/mfcc/train/' + file, 'r') as f:
            mfcc_train.append(f.read())

    # Write data to csv.
    with open('data/mfcc/test.csv', 'w') as f:
        for mfcc in mfcc_test:
            f.write(mfcc)

    with open('data/mfcc/train.csv', 'w') as f:
        for mfcc in mfcc_train:
            f.write(mfcc)
