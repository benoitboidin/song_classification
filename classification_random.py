import csv
import random

"""
Create a random classification for test.csv file.
"""


def load_test_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        test_data = list(reader)
    return test_data

def random_classification(test_data):
    for row in test_data:
        if row[0] == 'track_id':
            row.append('genre_id')
        else: 
            row.append(random.randint(1, 8))
    return test_data

def write_to_csv(test_data, output_filename='output_random.csv'):
    with open('output_random.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(test_data)


if __name__ == "__main__":
    filename = "test.csv"
    test_data = load_test_data(filename)
    test_data = random_classification(test_data)
    write_to_csv(test_data)