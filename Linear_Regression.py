import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Part 1: Plotting Data---
data = pd.read_csv("ex1data1.txt", header=None)

print(data.to_string())

plt.scatter(data[0], data[1])
plt.xticks(np.arange(5, 30, step=5))
plt.yticks(np.arange(-5, 30, step=5))
plt.title("Profit against Population")
plt.xlabel("Population of city (10,000)")
plt.ylabel("Profit (10,000)")
plt.show()


# --- Part 2: Compute Cost Function---
def computeCost(X, y, theta):
    m = len(y)
    prediction = np.dot(X, theta)  # X.dot(theta)
    sqaure_err = (prediction - y) ** 2

    return 1 / (2 * m) * np.sum(sqaure_err)


data_n = data.values
m = len(data_n)

# create feature table, with intercept
X = np.append(np.ones((m, 1)), data_n[:, 0].reshape(m, 1), axis=1)

y = data_n[:, 1].reshape(m, 1)

theta = np.zeros((2, 1))

print(computeCost(X, y, theta))


# --- Part 3: Gradient Descent---
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


theta, J_history = gradientDescent(X, y, 0.01, theta, 1500)

print("h(x) =" + str(round(theta[0, 0], 2)) + " + " + str(round(theta[1, 0], 2)) + "x1")

# --- Part 4: Visualisation---
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = computeCost(X, y, t)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot command
surface = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap="coolwarm")

# add colorbar
fig.colorbar(surface, shrink=0.5, aspect=5)

ax.set_xlabel("$\Theta_0$")
ax.set_ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")

plt.show()

# plot cost function curve

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost Function")
plt.title("Cost Function against Iterations")

plt.show()

# plot the best fit line
plt.scatter(data[0], data[1])
x_value = [x for x in range(25)]
y_value = [y * theta[1] + theta[0] for y in x_value]
plt.plot(x_value, y_value, color="r")

plt.xticks(np.arange(5, 30, step=5))
plt.yticks(np.arange(-5, 30, step=5))
plt.xlabel("Population of City (10,000)")
plt.ylabel("Profit (10,000)")
plt.title("Profit vs Population")
plt.show()


# prediction based on model

def predict(X, theta):
    predictions = np.dot(theta.transpose(), X)
    return predictions[0]


predict1 = predict(np.array([1, 3.5]), theta) * 10000

print("For population of 35,000 we predict profit of $" + str(round(predict1, 0)))

predict2 = predict(np.array([1, 7]), theta) * 10000
print("For population of 70,000 we predict profit of $" + str(round(predict2, 0)))
