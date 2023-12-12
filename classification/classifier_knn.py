"""
Classify music tracks using KNN.
Data is already split into train and test.
"""


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Read the data
def read_data(train_filename, test_filename):
    train = pd.read_csv(train_filename)
    test = pd.read_csv(test_filename)
    return train, test

# Train the model
def train_model(train):
    X = train.drop(['genre_id'], axis=1)
    y = train['genre_id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("kNN Accuracy:", accuracy_score(y_test, y_pred))
    print("kNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("kNN Classification Report:\n", classification_report(y_test, y_pred))
    return knn

# Predict the test data
def predict_test(test, knn):
    X = test
    y_pred = knn.predict(X)
    return y_pred

# Write the output to a csv file
def write_output(test, y_pred, output_filename):
    output = pd.DataFrame({'track_id': test['track_id'], 'genre_id': y_pred})
    # Add missing track IDs to the output file because messed up mfcc.
    output = pd.concat([output, pd.DataFrame({'track_id': [113025, 155298, 155306], 'genre_id': [1, 1, 1]})])
    output.to_csv(output_filename, index=False)

def main(train_mfcc_filename, test_mfcc_filename, output_knn_filename):
    print("\nkNN Reading data...")
    train, test = read_data(train_mfcc_filename, test_mfcc_filename)
    print("kNN Training model...")
    knn = train_model(train)
    print("kNN Predicting test data...")
    y_pred = predict_test(test, knn)
    write_output(test, y_pred, output_knn_filename)
    print("kNN Done!")

if __name__ == '__main__':
    main('data/train_mfcc.csv', 'data/test_mfcc.csv', 'data/output_knn.csv')