import pickle

# Train dataset : list of (id, genre, embed)
data = pickle.load(open('train_id_genres_vgg.pickle','rb'))

ids = [x[0] for x in data]
genres = [x[1] for x in data]
embed = [x[2] for x in data]

#one hot encoding
import tensorflow as tf
genres = [int(x)-1 for x in genres]
genres = tf.keras.utils.to_categorical(genres)

# Example
TRACK_NUMBER = 10
print("track n°", TRACK_NUMBER)
print("Song genres ", genres[TRACK_NUMBER])
print("Embed shape ", embed[TRACK_NUMBER].shape)
print("Embed ", embed[TRACK_NUMBER])

import numpy as np
x_train = np.array(embed)
y_train = np.array(genres)
print(x_train.shape, y_train.shape)

# Test dataset : list of (id, embed)
data = pickle.load(open('test_id_vgg.pickle','rb'))

tmp = [(x[0],x[1]) for x in data if len(x[1]) == 31]
ids = [x[0] for x in tmp]
embed = [x[1] for x in tmp]

# Example
TRACK_NUMBER = 10
print("track n°", TRACK_NUMBER)
print("Song id ", ids[TRACK_NUMBER])
print("Embed shape ", embed[TRACK_NUMBER].shape)
print("Embed ", embed[TRACK_NUMBER])

import numpy as np
embed = np.array(embed)
ids = np.array(ids)
print(embed.shape, ids.shape)