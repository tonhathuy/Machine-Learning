import pandas as pd
import matplotlib.pyplot as plt


datafame = pd.read_csv('data/Advertising.csv', header=0)
# print(datafame)
X = datafame.values[:, 2]
Y = datafame.values[:, 4]
# print(Y)

plt.scatter(X, Y, marker='o')
# plt.show()

def predict(new_radio, weight, bias):
    return weight*new_radio + bias

def cost_function(X, Y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error+=(Y[i] - (weight*X[i]+bias))**2

    return sum_error/n

def update_weight(X, Y, weight, bias, learning_rate):
    n = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2*X[i]*(Y[i] - (weight*X[i] + bias))
        bias_temp += -2*(Y[i] - (weight*X[i] + bias))
    weight -= (weight_temp/n)*learning_rate
    bias -= (bias_temp/n)*learning_rate

    return weight, bias

def train(X, Y, weight, bias, learning_rate, iter):
    coshis = []
    for i in range(iter):
        weight, bias = update_weight(X, Y, weight, bias, learning_rate)
        cost = cost_function(X, Y, weight, bias)
        coshis.append(cost)

    return weight, bias, coshis

weight, bias, cost = train(X, Y, 0.03, 0.0014, 0.01, 30)

print(weight, bias)
print(cost)

loop = [i for i in range(30)]
plt.plot(loop, cost)
plt.show()