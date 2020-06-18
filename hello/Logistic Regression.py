import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/data_classification.csv', header=None)
# print(data)

# print(data.values)
x_true = []
y_true = []
x_false = []
y_false = []

for item in data.values:
    if item[2] == 1.:
        x_true.append(item[0])
        y_true.append(item[1])
    else:
        x_false.append(item[0])
        y_false.append(item[1])

# plt.scatter(x_true, y_true, marker='o', c='b')
# plt.scatter(x_false, y_false, marker='o', c='r')
# plt.show()

# sigmoid function
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def phan_chia(p):
    return (p>=0.5)

def predict(features, weights):
    z = np.dot(features, weights)
    #3*1 + 4*6 + 6*8 = 75
    #[3, 4, 6]
    #[1, 6, 8]
    return sigmoid(z)

def cost_func(features, labels, weights):
    """
    : param features: (100x3)
    : param lables: (100x1) co 1 - 0 
    : param weights: (3x1)
    """
    n = len(labels)
    predictions = predict(features, weights)
    """
    predictions
    [0.8, 0.6, 0.8, 0.5]
    [1, 0, 1, 1]
    ma tran chuyen vi 
    [[1],
    [0],
    [1],
    [1],
    ]
    """
    cost_class1 = -labels.np.log(predictions)
    cost_class2 = -(1 - labels).np.log(1 - predictions)
    cost = cost_class1 + cost_class2
    return cost.sum()/n

def update_weight(features, labels, weights, learning_rate):
    """
    : param features: (100x3)
    : param lables: (100x1) co 1 - 0 
    : param weights: (3x1)
    : param learning rate: float

    gradient desent 
    s'(z) = s(z)(1-s(z))
    -> C' = x( s(z) -y )

    """
    n = len(labels)
    predictions = predict(features, weights)
    gd = np.dot(features.T, (predictions - labels)
    gd = gd/n
    gd = gd*learning_rate
    weights = weights - gd
    return weights

def train(features, labels, weights, learning_rate, iter):
    cost_his = []
     for i in range(iter):
         weights = update_weight(features, labels, weights, learning_rate)
         cost = cost_func(features, labels, weights)
        cost_his.append(cost)
    
    return weights, cost_his
N,d = data.values.shape
# print(d)
labels = data.values[:, 2:3]
# print(labels)
features = data.values[:, 0:2]
# x = np.hstack((np.ones((N, 1)), x))
features = np.hstack((np.ones((N,1)),features))
# print(features)
# print(data.values)
weights = np.array([0., 0.1, 0.1])
# print(weights)
weight, cost = train(features, labels, weights, learning_rate = 0.01, 30)

print(cost)