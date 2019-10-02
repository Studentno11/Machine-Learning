# Single feature.
import numpy as np

x = np.random.rand(100)
y = np.random.randint(0,2,100)

def compute_error(y, y_pred):
    # - 1/m sum(y log(H) + (1-y) log(1-H))
    error = 0
    for i in range(len(y)):
        error += y[i] * np.log(y_pred[i]) + (1 - y[i]) * np.log(1 - y_pred[i])
    
    error = - (error / len(y))
    return error

def grad(x, y, w, b):
    j = 0
    dw = 0
    db = 0
    for i in range(len(y)):
        z = w * x[i] + b
        a = 1 / (1 + np.exp(-z))
        j += - (y[i] * np.log(a) + (1 - y[i]) * np.log(1 - a))
        dz = a - y[i]
        dw += x[i] * dz
        db += dz
    m = len(y)
    j /= m
    dw /= m
    db /= m
    return dw, db, j

def grad2(x, y, w, b):
    learning_rate = 0.1
    for i in range(5000):
        z = x * w + b
        a = 1 / (1 + np.exp(-z))
        w -= learning_rate * np.sum((a - y) * x) / len(x)
        b -= learning_rate * np.sum(a - y) / len(x)
        error = - np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / len(x)
        if i % 50 == 0:
            print("grad2: ", error)

w = 1
b = 0

# compute gradient formular 1.
learning_rate = 0.1
for i in range(5000):
    dw, db, j = grad(x, y, w, b)
    w -= learning_rate * dw
    b -= learning_rate * db
    if i % 50 == 0:
        print(j)

# Compute gradient formular 2.
grad2(x, y, w, b)

    
    