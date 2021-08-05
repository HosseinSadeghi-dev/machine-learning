import numpy as np
import matplotlib.pyplot as plt

N = 1000
std = 1

melon_width = np.random.normal(30, std, N)
melon_height = np.random.normal(10, std, N)
melon_weight = np.random.normal(7, std, N)

balloon_width = np.random.normal(10, std, N)
balloon_height = np.random.normal(20, std, N)
balloon_weight = np.random.normal(0.35, std - 1, N)

width = np.concatenate((melon_width, balloon_width))
height = np.concatenate((melon_height, balloon_height))
weight = np.concatenate((melon_weight, balloon_weight))

X = np.array([width, height, weight]).T
melon_label = np.zeros(N, dtype='int')
balloon_label = np.ones(N, dtype='int')

Y = np.concatenate((melon_label, balloon_label))

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(X[:N, 0], X[:N, 1], X[:N, 2], c='red')
ax.scatter(X[N:, 0], X[N:, 1], X[N:, 2], c='green')

ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('weight')
