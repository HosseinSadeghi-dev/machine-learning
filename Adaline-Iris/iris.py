import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from KNN.KNearestNeighbour import MyKNearestNeighbors
from Adaline.Adaline import AdalineRegressor

IRIS = load_iris()
X = []
Y = []

for i, t in enumerate(IRIS.target):
    if t == 2:
        break
    X.append(IRIS.data[i])
    Y.append(IRIS.target[i])

X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2)

knn = MyKNearestNeighbors(5)
knn.fit(X_train, Y_train)

print('k= 5  Accuracy=', knn.evaluate(X_test, Y_test))

model = AdalineRegressor()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
result = model.evaluation(X_test, Y_test)
print('evaluation= ', result[0], ' Accuracy= ', result[1])
