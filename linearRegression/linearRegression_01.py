# https://jermwatt.github.io/machine_learning_refined/notes/5_Linear_regression/5_6_Multi.html
# https://github.com/jermwatt/machine_learning_refined/tree/gh-pages/mlrefined_exercises/ed_2/chapter_5
# https://jermwatt.github.io/machine_learning_refined/notes/5_Linear_regression/5_6_Multi.html
# https://github.com/jermwatt/machine_learning_refined/blob/gh-pages/notes/5_Linear_regression/5_6_Multi.ipynb

# https://donaldpinckney.com/books/pytorch/book/ch2-linreg/2018-03-21-multi-variable.html
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
# https://github.com/jermwatt/machine_learning_refined
# https://pytorch.org/docs/stable/tensors.html
# https://donaldpinckney.com/books/pytorch/book/ch2-linreg/2018-03-21-multi-variable.html
# https://pytorch.org/docs/stable/optim.html
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/

# musi se to preskalovat jinak bude vzdy loss vychazet obrovsky 

# %% libs 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import networkx as nx
from sklearn.model_selection import train_test_split
import torch 
import torch as th
import torch.nn as nn
import numpy as np
# import sys
# # compute linear combination of input points
# def model(x,w):
#     a = w[0] + np.dot(x.T,w[1:])
#     return a.T
# # an implementation of the least squares cost function for linear regression
# def least_squares(w):    
#     # compute the least squares cost
#     cost = np.sum((model(x,w) - y)**2)
#     return cost/float(np.size(y))
# %%
file_CF  = '../rescomp/dsallCF2_01.csv'
file_RF  = '../rescomp/dsallRF2_01.csv'
file_S   = '../rescomp/dsallS_01.csv'
G_adj    = '../preping/B2.adjlist'  # # grapooh defined by adjacency list 
G_ml     = '../preping/B2.graphml'  
# %% reading data
y = pd.read_csv(file_S).T
y = y.drop(index='Unnamed: 0')
x_CF = pd.read_csv(file_CF)
x_CF = x_CF.T 
x_RF = pd.read_csv(file_RF)
x_RF = x_RF.T
d= pd.concat([x_CF,x_RF], axis = 1)


# X = d.iloc[1:,[2,8]] #
# y = d.iloc[1:,[2,3]] #

X = d.iloc[1:,[2,8,9]] #
y = d.iloc[1:,:] #

# X = d.iloc[1:,[2,8,9]] #
G = nx.read_graphml(G_ml)
seeding_magic_number = 42  
selectEach = 5
x= X.iloc[::selectEach, :]
y= y.iloc[::selectEach, :]
x = th.tensor(x.values, dtype=torch.float)
y = th.tensor(y.values, dtype=torch.float)


# %%

# https://pytorch.org/docs/stable/optim.html
# from mlrefined_libraries import superlearn_library as superlearn
# from mlrefined_libraries import superlearn_library as superlearn
# from mlrefined_libraries import math_optimization_library as optlib
# from mlrefined_libraries import math_optimization_library as optlib
## This code cell will not be shown in the HTML version of this notebook
# setup and run optimization
# g = least_squares; 
# w = 0.1*np.random.randn(3,2)
# max_its = 200;
# alpha_choice = 1;
# weight_history,cost_history = optimizers.gradient_descent(g,alpha_choice,max_its,w)

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# %%
# plot history
# static_plotter.plot_cost_histories([cost_history],start = 0,points = False,labels = ['run 1'])


### Load the data

# First we load the entire CSV file into an m x 3
# D = torch.tensor(pd.read_csv("linreg-multi-synthetic-2.csv", header=None).values, dtype=torch.float)

# and then transpose it
x_dataset = x.T
y_dataset = y.T
# x_dataset = x
# y_dataset = y

# And make a convenient variable to remember the number of input columns
n = 3


### Model definition ###

# First we define the trainable parameters A and b 
A = torch.randn((1, n), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Then we define the prediction model
def model(x_input):
    return A.mm(x_input) + b


### Loss function definition ###

def loss(y_predicted, y_target):
    return ((y_predicted - y_target)**2).sum()

### Training the model ###

# Setup the optimizer object, so it optimizes a and b.
optimizer = optim.Adam([A, b], lr=0.1)
# %%

lossnp = np.zeros([20000])
# %%
# Main optimization loop
for t in range(2000):
    # Set the gradients to 0.
    optimizer.zero_grad()
    # Compute the current predicted y's from x_dataset
    y_predicted = model(x_dataset)
    # See how far off the prediction is
    current_loss = loss(y_predicted, y_dataset)
    # Compute the gradient of the loss with respect to A and b.
    current_loss.backward()
    # Update A and b accordingly.
    optimizer.step()
    print(f"t = {t}, loss = {current_loss}")
    lossnp[t] = current_loss
    # print(f"t = {t}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")
    
# %%
plt.plot(lossnp)
# %%
# 	2	3	4
# I8	= th.tensor([-40.0	29.979	9.8267

y_predicted = model(x_dataset)
# %%
# https://www.geeksforgeeks.org/linear-regression-using-pytorch/
import torch
from torch.autograd import Variable

# x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
# y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
x_data = x.T
y_data = y.T

class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(3, 2) # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

# our model
our_model = LinearRegressionModel()

criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

for epoch in range(500):

	# Forward pass: Compute predicted y by passing
	# x to the model
	pred_y = our_model(x_data)

	# Compute and print loss
	loss = criterion(pred_y, y_data)

	# Zero gradients, perform a backward pass,
	# and update the weights.
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('epoch {}, loss {}'.format(epoch, loss.item()))

new_var = Variable(torch.Tensor([[4.0]]))
pred_y = our_model(new_var)
print("predict (after training)", 4, our_model(new_var).item())

# %%%
# https://www.kaggle.com/code/aakashns/pytorch-basics-linear-regression-from-scratch
# %%
# https://www.projectpro.io/recipes/do-linear-regression-pytorch
# https://www.kaggle.com/code/aakashns/pytorch-basics-linear-regression-from-scratch
