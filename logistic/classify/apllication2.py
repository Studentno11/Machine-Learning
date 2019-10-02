# Kittipat Phongsak
# Machine learning
# Logistic regression
""" Used The Canny edge and Regularized fixed Overflow """

import pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pickle_in = open("features.pickle", "rb")
features = pickle.load(pickle_in)

pickle_in = open("labels.pickle", "rb")
labels = pickle.load(pickle_in)

IMG_SIZE = 64
X = []
for feature in features:
    feature = cv.Canny(feature, 100, 200)
    X.append(cv.resize(feature, (IMG_SIZE, IMG_SIZE)))


X = np.array(X)
Y = np.array(labels)

X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T 
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train_flatten = X_train_flatten/255.
X_test_flatten = X_test_flatten/255.

w = np.zeros((X_train_flatten.shape[0], 1))
b = 0
nums_iterations = 3000
learning_rate = 0.03
m = X_train_flatten.shape[1]
costs = []
lamda = 1000
for i in range(nums_iterations):
    z = np.dot(w.T, X_train_flatten) + b
    A = 1 / (1 + np.exp(-z))
    cost = (- np.sum(y_train * np.log(A) + (1 - y_train) * np.log(1 - np.log(A))) / m) + ((lamda * np.sum(w**2)) / (2 * m))
    dw = np.dot(X_train_flatten, (A - y_train).T) / m
    db = np.sum(A - y_train) / m
    w = w - learning_rate * (dw + ((lamda * w) / m))
    b = b - learning_rate * db
    if i % 100 == 0:
        costs.append(cost)  
        print(f"nums {i} cost: {cost}.")

z_train = np.dot(w.T, X_train_flatten) + b
A_train = 1 / (1 + np.exp(-z_train))
y_train_predict = np.zeros((1, m))

z_test = np.dot(w.T, X_test_flatten) + b
A_test = 1 / (1 + np.exp(-z_test))
y_test_predict = np.zeros((1, X_test_flatten.shape[1]))

for i in range(A_train.shape[1]):
    y_train_predict[0, i] = 0 if A_train[0, i] <= 0.5 else 1

for i in range(A_test.shape[1]):
    y_test_predict[0, i] = 0 if A_test[0, i] <= 0.5 else 1
print("train accuracy: {} %".format(100 - np.mean(np.abs(y_train_predict - y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_test_predict - y_test)) * 100))

costs = np.squeeze(costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iteration (per hundreds)')
plt.title("learning rate =" + str(learning_rate))
plt.show()
