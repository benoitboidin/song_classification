"""
Classifiy music genre using Keras.
Data is already split into train and test.
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, GRU
from keras.callbacks import Callback, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def train_model(train, train_labels):

    X = train
    y = train_labels

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    # encoder = LabelEncoder()
    # encoder.fit(Y_train)

    # Y_train = encoder.transform(Y_train).reshape([len(Y_train), 1])

    # encoder = LabelEncoder()
    # encoder.fit(Y_test)

    # Y_test = encoder.transform(Y_test).reshape([len(Y_test), 1])

    #Initiating the model as Sequential
    model = Sequential()

    #Adding the CNN layers along with some drop outs and maxpooling
    model.add(Conv2D(64, 2, activation = 'relu', input_shape = (X_train.shape[1:])))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, 2, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(256, 2, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (4,4)))
    model.add(Dropout(0.1))
    model.add(Conv2D(512, 2, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (8,8)))
    model.add(Dropout(0.1))

    #flattening the data to be passed to a dense layer
    model.add(Flatten())

    #Adding the dense layers
    model.add(Dense(2048, activation = 'relu'))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))

    #final output layer with 10 predictions to be made
    model.add(Dense(10, activation = 'softmax'))

    '''
    Optimizer = Adam
    Loss = Sparse Categorical CrossEntropy
    '''
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    #fitting the model
    model.fit(X_train, Y_train, epochs = 100, validation_data = (X_test, Y_test), callbacks = [EarlyStopping(monitor='val_loss', patience=2)])

    # Display accuray
    print('Model is fitted: ' + str(model.built))
    print('Model params:')
    print(model.get_config())
    print('Model train accuracy: ' + str(model.evaluate(X_test, Y_test)[1]))

    return model

def main(train_filename, test_filename, output_filename):

    print("Reading data...")
    train = pd.read_csv(train_filename)
    train_labels = pd.read_csv('data/train.csv').genre_id
    test = pd.read_csv(test_filename)

    print("Training model...")
    model = train_model(train, train_labels)

    # print("Predicting test data...")
    # y_pred = predict_test(test, model)

    # print("Writing output...")
    # write_output(test, y_pred, output_filename)

    print("Done.")


if __name__ == '__main__': 
    main('data/train_features.csv', 
         'data/test_features.csv', 
         'output/output_keras.csv')