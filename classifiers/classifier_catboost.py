"""
Classify music tracks using CatBoost.
Data is already split into train and test.
"""


import pandas as pd
import numpy as np
import catboost as cbt
from sklearn.model_selection import train_test_split


# Train the model
def train_model(train, train_labels):

    X = train
    y = train_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = cbt.CatBoostClassifier(
        iterations=700,
        learning_rate=0.2,
        depth=6,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        logging_level='Silent'
    )
    # train the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)

    # Display accuray
    print('Model is fitted: ' + str(model.is_fitted()))
    print('Model params:')
    print(model.get_params())
    print('Model train accuracy: ' + str(model.score(X_test, y_test)))

    return model

# Predict the test data
def predict_test(test, catboost):
    X = test
    y = catboost.predict(X)
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
    catboost = train_model(train, train_labels)

    print("Predicting test data...")
    y_pred = predict_test(test, catboost)
    print(y_pred)
    write_output(test, y_pred, output_filename)

    print("Done!")


if __name__ == '__main__':
    main('data/train_features.csv', 
         'data/test_features.csv', 
         'output/output_catboost.csv')
