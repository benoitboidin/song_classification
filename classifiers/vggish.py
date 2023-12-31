import tensorflow as tf
import numpy as np
import pickle


def get_data(train_filename='train_id_genres_vgg.pickle', test_filename='test_id_vgg.pickle'):
    # Train dataset : list of (id, genre, embed)
    data = pickle.load(open(train_filename,'rb'))

    ids = [x[0] for x in data]
    genres = [x[1] for x in data]
    embed = [x[2] for x in data]

    #one hot encoding
    genres = [int(x)-1 for x in genres]
    genres = tf.keras.utils.to_categorical(genres)

    # Example
    TRACK_NUMBER = 10
    # print('\n TRAIN DATASET')
    # print("track n°", TRACK_NUMBER)
    # print("Song genres ", genres[TRACK_NUMBER])
    # print("Embed shape ", embed[TRACK_NUMBER].shape)
    # print("Embed ", embed[TRACK_NUMBER])

    x_train = np.array(embed)
    y_train = np.array(genres)
    print(x_train.shape, y_train.shape)

    # Test dataset : list of (id, embed)
    data = pickle.load(open(test_filename,'rb'))

    tmp = [(x[0],x[1]) for x in data if len(x[1]) == 31]
    ids = [x[0] for x in tmp]
    embed = [x[1] for x in tmp]

    # Example
    TRACK_NUMBER = 10
    # print('\n TEST DATASET')
    # print("track n°", TRACK_NUMBER)
    # print("Song id ", ids[TRACK_NUMBER])
    # print("Embed shape ", embed[TRACK_NUMBER].shape)
    # print("Embed ", embed[TRACK_NUMBER])

    embed = np.array(embed)
    ids = np.array(ids)
    print(embed.shape, ids.shape)

    return x_train, y_train, embed, ids


if __name__ == '__main__': 
    print('Start...')

    x_train, y_train, x_test, ids = get_data('data/train_id_genres_vgg.pickle', 
                                              'data/test_id_vgg.pickle')

    print('------------------')
    print(x_train.shape)
    print('------------------')
    # print(y_train)
    # print('------------------')
    print(x_test.shape)
    print('------------------')
    # print(ids)
    # print('------------------')
