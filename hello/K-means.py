from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 5
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
print(X0)

# Gộp 3 ma trân X0, X1, X2 thành X theo chiều ngang
X = np.concatenate((X0, X1, X2), axis = 0)
K = 3
# Khởi tạo vector label có length = 3N, N phần tử đầu là 0, N phần tử ở giữa là 1, N phần tử cuối là 2
# asarray: Convert a list into an array
original_label = np.asarray([0]*N + [1]*N + [2]*N).T
print(original_label)
# def kmeans_display(X, label):
#     K = np.amax(label) + 1
#     X0 = X[label == 0, :]
#     X1 = X[label == 1, :]
#     X2 = X[label == 2, :]
    
#     plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
#     plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
#     plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

#     plt.axis('equal')
#     plt.plot()
#     plt.show()
    
# kmeans_display(X, original_label)

