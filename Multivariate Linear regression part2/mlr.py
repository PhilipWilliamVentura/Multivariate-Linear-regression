import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

my_data = pd.read_csv("/Users/hello./Documents/GitHub/Linear-regression/Multivariate Linear regression part2/home.txt", names=["size", "bedroom", "price"])

#NORMALIZATION
my_data = (my_data - my_data.mean()) / my_data.std()

#CREATE MATIXES
X = my_data.iloc[:,0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X), axis=1)

y = my_data.iloc[:,2:3].values
theta = np.zeros([1,3])

#HYPER PARAMETERS
alpha = 0.01
iters = 10000

def find_cost(X, y, theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2*len(X))

def gradient_descent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = find_cost(X, y, theta)

    return theta, cost

#PRINT COST
g,cost=gradient_descent(X, y, theta, iters, alpha)
print(g)
finalCost = find_cost(X, y, g)
print(finalCost)

#COST PLOT
fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost, "r")
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.set_title("Error vs Training Epoch")
plt.show()

#3D SURFACE PLOT (Visualization of regression plane)
size = my_data["size"]
bedroom = my_data["bedroom"]
price = my_data["price"]

size_range = np.linspace(size.min(), size.max(), 100)
bedroom_range = np.linspace(bedroom.min(), bedroom.max(), 100)
size_grid, bedroom_grid = np.meshgrid(size_range, bedroom_range)

price_pred = g[0,0] + g[0,1]*size_grid + g[0,2]*bedroom_grid

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(size, bedroom, price, color="red", label="Actual Prices")
ax.plot_surface(size_grid, bedroom_grid, price_pred, alpha=0.5, color='blue')

ax.set_xlabel("Size (sq ft)")
ax.set_ylabel("Number of Bedrooms")
ax.set_zlabel("Price ($)")
ax.set_title("Multivariate Linear Regression Model")

plt.show()