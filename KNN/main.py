import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from KNearestNeighbour import MyKNearestNeighbors

N = 1000
std = 0.5

banana_width = np.random.normal(4, std, N)
banana_length = np.random.normal(8, std, N)

apple_width = np.random.normal(6, std, N)
apple_length = np.random.normal(6, std + 0.1, N)

width = np.concatenate((banana_width, apple_width))
length = np.concatenate((banana_length, apple_length))

banana_label = np.zeros(N, dtype='int')
apple_label = np.ones(N, dtype='int')

X = np.array([width, length]).T
Y = np.concatenate((banana_label, apple_label))

# test data
test_banana_width = np.random.normal(4, std, 100)
test_banana_length = np.random.normal(8, std, 100)

test_apple_width = np.random.normal(6, std, 100)
test_apple_length = np.random.normal(6, std + 0.1, 100)

test_width = np.concatenate((test_banana_width, test_apple_width))
test_length = np.concatenate((test_banana_length, test_apple_length))

test_banana_label = np.zeros(100, dtype='int')
test_apple_label = np.ones(100, dtype='int')

X_test = np.array([test_width, test_length]).T
Y_test = np.concatenate((test_banana_label, test_apple_label))

plt.figure(figsize=(12, 12))

plt.scatter(X[0:N, 1], X[0:N, 0], c='orange', label='banana')
plt.scatter(X[N:2 * N, 1], X[N:2 * N, 0], c='red', label='apple')

plt.xlabel('Length')
plt.ylabel('Width')

fruits = {
    0: 'موز',
    1: 'سیب'
}

knn = MyKNearestNeighbors(5)
knn.fit(X, Y)
knn.evaluate(X_test, Y_test)

knc = KNeighborsClassifier(5)
knc.fit(X, Y)
print(knc.score(X_test, Y_test))
