import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv("home.txt", names=["size", "bedroom", "price"])

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
iters = 1000
