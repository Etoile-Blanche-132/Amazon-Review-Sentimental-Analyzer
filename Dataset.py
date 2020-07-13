import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

MAX_LENGTH = 30

class DATA():
    def __init__(self):
        with open('data.json', 'r') as f:
            data = json.load(f)

        X = data['reviewText']
        y = data['overall']

        X = self.VectorReshaper(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        self.setSize = len(X_train)
        self.inputLength = len(X_train[0])

    def VectorReshaper(self, X):
        X = pad_sequences(X, maxlen = MAX_LENGTH)

        return X

if __name__ == "__main__":
    data = DATA()
