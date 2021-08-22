import numpy as np


class MyKNearestNeighbors:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.number_class = None

    # train
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.number_class = len(np.unique(y_train))

    def nearest_neighbors(self, x_test):
        distance = np.sqrt(np.sum((x_test - self.x_train) ** 2, axis=1))
        near_neighbor = np.argsort(distance)[0:self.k]
        return near_neighbor

    # test
    def predict(self, x_test):
        near_neighbor = self.nearest_neighbors(x_test)
        return np.argmax(np.bincount(self.y_train[near_neighbor]))

    def evaluate(self, x_test, y_test):
        y_pred = []
        for i in range(len(x_test)):
            y = self.predict(x_test[i])
            y_pred.append(y)
        not_correct = abs(sum(y_pred - y_test))
        print((((len(y_pred) - not_correct) * 100) / len(y_pred)) / 100)
        return (y_pred, y_test), ((((len(y_pred) - not_correct) * 100) / len(y_pred)) / 100)
