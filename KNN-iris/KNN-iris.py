import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from KNN.KNearestNeighbour import MyKNearestNeighbors

IRIS = load_iris()
X = IRIS.data
Y = IRIS.target
N = (len(X) // len(IRIS.target_names))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

plt.scatter(X[0:N, 1], X[0:N, 0], c='red')
plt.scatter(X[N:2 * N, 1], X[N:2 * N, 0], c='green')
plt.scatter(X[2 * N:3 * N, 1], X[2 * N:3 * N, 0], c='blue')

K = [1, 3, 5, 7, 9, 11, 13, 15]
result = []

for i in K:
    knn = MyKNearestNeighbors(i)
    knn.fit(X_train, Y_train)
    result.append([i, knn.evaluate(X_test, Y_test)[1]])

for j in result:
    print('k=', j[0], '  Accuracy=', j[1])

result = np.array(result)

plt.bar(result[:, 0], result[:, 1])
##########################################
knn5 = MyKNearestNeighbors(5)
knn5.fit(X_train, Y_train)
res_y_pred = knn5.evaluate(X_test, Y_test)[0][0]
res_y_test = knn5.evaluate(X_test, Y_test)[0][1]

set_set = 0
set_ver = 0
set_vir = 0
vir_set = 0
vir_ver = 0
vir_vir = 0
ver_set = 0
ver_ver = 0
ver_vir = 0

for i in range(len(res_y_pred)):
    if res_y_pred[i] == 0 and res_y_test[i] == 0:
        set_set += 1
    elif res_y_pred[i] == 1 and res_y_test[i] == 1:
        ver_ver += 1
    elif res_y_pred[i] == 2 and res_y_test[i] == 2:
        vir_vir += 1
    elif res_y_pred[i] == 0 and res_y_test[i] == 1:
        set_ver += 1
    elif res_y_pred[i] == 0 and res_y_test[i] == 2:
        set_vir += 1
    elif res_y_pred[i] == 2 and res_y_test[i] == 0:
        vir_set += 1
    elif res_y_pred[i] == 2 and res_y_test[i] == 1:
        vir_ver += 1
    elif res_y_pred[i] == 1 and res_y_test[i] == 0:
        ver_set += 1
    else:
        ver_vir += 1

result = np.array([[set_set, set_ver, set_vir],
                   [ver_set, ver_ver, ver_vir],
                   [vir_set, vir_ver, vir_vir]])

df_cm = pd.DataFrame(result, index=[i for i in ['setosa', 'versicolor', 'virginica']],
                     columns=[i for i in ['setosa', 'versicolor', 'virginica']])
plt.figure(figsize=(10, 7))
seaborn.heatmap(df_cm, annot=True)
