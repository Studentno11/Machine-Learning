# Multiple features.
import numpy as np

feature = np.random.rand(3, 100) # Shape(3, 100) three dimensions.
label = np.random.randint(0, 2, 100) # Shape(100) one dimension.

# Initialize parameter
weight = {}
bias = 0
for i in range(feature.shape[0]):
    weight["W" + str(i)] = np.random.rand()
print("parameter start: ", weight, bias)

# Test Hypothesis.
z = 0
for i in range(feature.shape[0]):
    z += feature[i] * weight["W" + str(i)]
z += bias
a = 1 / (1 + np.exp(-z))
# print(a)

def optimize(x, y, w, b):
    learning_rate = 0.1
    nums_iters = 2500
    for i in range(nums_iters + 1):
        z = 0
        for j in range(x.shape[0]):
            z += x[j] * w["W" + str(j)]
        z += b
        a = 1 / (1 + np.exp(-z))
        cost = - np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / len(y)
        if i % 500 == 0:
            print(f"{i}: {cost}")
        
        if i == nums_iters:
            break

        for j in range(x.shape[0]):
            w["W" + str(j)] -= learning_rate * np.sum((a - y) * x[j]) / len(x[j])
        b -= learning_rate * np.sum(a - y) / len(y)
    print("Weight: ", w)
    print("Bias: ", b)

def optimize2(x, y, weight, bias):
    learning_rate = 0.1
    nums_iters = 2500
    for i in range(nums_iters + 1):
        z = 0
        for j in range(x.shape[0]):
            z += x[j] * weight["W" + str(j)]
        z += bias
        a = 1 / (1 + np.exp(-z))
        cost = - np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / len(y)
        if i % 500 == 0:
            print(f"{i}: {cost}")
        dz = a - y

        if i == nums_iters:
            break
        for j in range(x.shape[0]):
            weight["W" + str(j)] -= learning_rate * np.sum(dz * x[j]) / len(x[j])
        bias -= learning_rate * np.sum(dz) / len(y)
    print("Weight2: ", weight)
    print("Bias2: ", bias)

weight1 = weight.copy()
weight2 = weight.copy()
optimize(feature, label, weight1, bias)
optimize2(feature, label, weight2, bias)
print("parameter end: ", weight, bias)