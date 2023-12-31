"""
Classify music tracks using TensorFlow deep learning model.
Data is already split into train and test.

Notes:
Scores are too low. Need to improve the model.
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

import vggish


# Train the model
def train_model(train, train_labels):

        # Shape of the data: (3995, 31, 128)
        X = train
        y = train_labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # # Reshape the data
        X_train = X_train.reshape(-1, 31*128)
        X_test = X_test.reshape(-1, 31*128)

        # # # Normalize the data
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # # Reshape the labels
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # # One hot encode the labels
        encoder = LabelBinarizer()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.fit_transform(y_test)

        y_train = y_train.reshape(-1, 8)
        y_test = y_test.reshape(-1, 8)

        # # # Define the model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax'),
        ])

        # # One hot encode the labels
        # # encoder = LabelBinarizer()
        # # y_train = encoder.fit_transform(y_train)
        # # y_test = encoder.fit_transform(y_test)

        # # y_train = y_train.reshape(-1, 8)
        # # y_test = y_test.reshape(-1, 8)

        # # # Define the model
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(256, activation='relu'),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dense(8, activation='softmax'),
        # ])

        # Compile the model
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, batch_size=32, epochs=40)

        # Display accuray
        print('Model is fitted: ' + str(model.built))
        print('Model params:')
        # print(model.get_config())
        print('Model train accuracy: ' + str(model.evaluate(X_test, y_test)[1]))

        return model

# Predict test data
def predict_test(test, model):
    X = test
    # Predict test data
    y_pred = model.predict(X)
    # Convert predictions to labels
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred

# Write output
def write_output(test, y_pred, output_filename):

    # Write output
    output = pd.DataFrame({'track_id': test.track_id, 'genre_id': y_pred})
    output.to_csv(output_filename, index=False)

def main(train_filename, test_filename, output_filename):

    print("Reading data...")
    # train = pd.read_csv(train_filename)
    # train_labels = pd.read_csv('data/train.csv').genre_id
    test = pd.read_csv(test_filename)
    x_train, y_train, x_test, _ = vggish.get_data('data/train_id_genres_vgg.pickle', 
                                              'data/test_id_vgg.pickle')

    print("\nTraining model...")
    model = train_model(x_train, y_train)

    # print("\nPredicting test data...")
    # y_pred = predict_test(x_test, model)

    # print("\nWriting output...")
    # write_output(test, y_pred, output_filename)

    # print("Done.")


if __name__ == '__main__': 
    main('data/train_features.csv', 
         'data/test_features.csv', 
         'data/output_tensorflow.csv')
