"""
Classify music tracks using CatBoost.
Data is already split into train and test.
"""

import pandas as pd
import catboost as cbt


# Read the data
def read_data(train_filename, test_filename):
    train = pd.read_csv(train_filename)
    test = pd.read_csv(test_filename)
    return train, test

# Train the model
def train_model(train):

    X = train.drop(['genre_id'], axis=1)
    y = train['genre_id']

    model = cbt.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        logging_level='Silent'
    )
    # train the model
    return model.fit(X, y)

# Predict the test data
def predict_test(test, catboost):
    X = test
    y_pred = catboost.predict(X)
    return y_pred

# Write the output to a csv file
def write_output(test, y_pred, output_filename):
    output = pd.DataFrame({'track_id': test['track_id'], 'genre_id': y_pred})
    output.to_csv('output_catboost.csv', index=False)

def main(train_filename, test_filename, output_filename):
    train_filename = 'data/train_mfcc.csv'
    test_filename = 'data/test_mfcc.csv'
    output_filename = 'output/output_catboost.csv'

    print("Reading data...")
    train, test = read_data(train_filename, test_filename)
    
    print("Training model...")
    catboost = train_model(train)

    print("Predicting test data...")
    y_pred = predict_test(test, catboost)
    write_output(test, y_pred, output_filename)

    print("Done!")


if __name__ == '__main__':
    main('data/train_mfcc.csv', 'data/test_mfcc.csv', 'data/output_catboost.csv')