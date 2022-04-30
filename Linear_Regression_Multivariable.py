import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Part 1 : Plotting and visualizing data

data2 = pd.read_csv("ex1data2.txt", header=None)

# we have 2 features, create 2 subplots in a figure

fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=2)

axes[0].scatter(data2[0], data2[2], color="b")
axes[0].set_xlabel("Size in square feet")
axes[0].set_ylabel("Prices")
axes[0].set_title("Cost against size of square feet")

axes[1].scatter(data2[1], data2[2], color="r")
axes[1].set_xlabel("Number of Bedrooms")
axes[1].set_ylabel("Prices")

axes[1].set_xticks(np.arange(1, 6, step=1))
axes[1].set_title("Cost against Number of Bedrooms")

plt.show()


# Part 2: feature Normalization

def featureNormalization(X):
    mean = np.mean(X, axis=0)
    standard = np.std(X, axis=0)

    X_norm = (X - mean) / standard

    return X_norm, mean, standard


data = data2.values

m = len(data)

# extract features
x = data[:, 0:2].reshape(m, 2)  # why 0:2??

norm_X, mean_X, std_X = featureNormalization(x)

norm_X = np.append(np.ones((m, 1)), norm_X, axis=1)

y = data[:, -1].reshape(m, 1)

# initialize theta

theta = np.zeros((3, 1))


# Part 3: Gradient Descent

def computeCost(X, y, theta):
    m = len(y)
    prediction = np.dot(X, theta)
    sqaure_error = (prediction - y) ** 2

    return 1 / 2 * m * np.sum(sqaure_error)


def gradientDescent(X, y, alpha, theta, iter):
    m = len(y)
    J_history = []

    for i in range(iter):
        prediction = np.dot(X, theta)
        error = np.dot(X.transpose(), (prediction - y))
        descent = alpha * 1 / m * error
        theta = theta - descent
        J_history.append(computeCost(X, y, theta))

    return theta, J_history


theta, J_history = gradientDescent(norm_X, y, 0.01, theta, 400)

print("h(x) =" + str(round(theta[0, 0], 2)) + "+" + str(round(theta[1, 0], 2)) + "x1" "+" + str(
    round(theta[2, 0], 2)) + "x2")  # must call theta row and column

# Plot Cost function

plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("Cost function")
plt.title("Cost function using Gradient Descent")
plt.show()


# Prediction using the model
def predict(X, theta):
    predictions = np.dot(theta.transpose(), X)
    return predictions[0]


sample = featureNormalization(np.array([1650, 3]))[0]

sample = np.append(np.ones(1), sample)

predict1 = predict(sample, theta)

print("For the house size of 1650 with 3 bedroom, the predicted price is $" + str(round(predict1, 0)))


