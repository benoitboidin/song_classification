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


# Train the model
def train_model(train, train_labels):

        X = train
        y = train_labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # One hot encode the labels
        encoder = LabelBinarizer()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.fit_transform(y_test)

        # Define the model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                    loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=10)

        # Display accuray
        print('Model is fitted: ' + str(model.built))
        print('Model params:')
        print(model.get_config())
        print('Model train accuracy: ' + str(model.evaluate(X_test, y_test)))

        return model

# Predict the test data
def predict_test(test, model):
    X = test
    y = model.predict(X)
    return np.hstack(y)

# Write the output to a csv file
def write_output(test, y_pred, output_filename):
    output = pd.DataFrame({'track_id': test['track_id'], 
                           'genre_id': y_pred})
    output.to_csv(output_filename, index=False)

def main(train_filename, test_filename, output_filename):

    print("Reading data...")
    train = pd.read_csv(train_filename)
    train_labels = pd.read_csv('data/train.csv').genre_id
    test = pd.read_csv(test_filename)

    print("Training model...")
    model = train_model(train, train_labels)

    print("Predicting test data...")
    y_pred = predict_test(test, model)

    print("Writing output...")
    write_output(test, y_pred, output_filename)

    print("Done.")


if __name__ == '__main__': 
    main('data/train_features.csv', 
         'data/test_features.csv', 
         'data/output_tensorflow.csv')
