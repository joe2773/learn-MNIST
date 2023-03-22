import numpy as np
from numpy import ndarray
from model import Model
import pickle
learning_rate = 0.01
num_epochs = 2

with open('mnist_data.pkl', 'rb') as f:
    mnist = pickle.load(f)
inputs = mnist['data'][:1000] 
targets = mnist['target'][:1000].astype(np.uint8)
param_info = np.array([
    {'operationName' : 'linearRegression','params' : np.random.rand(inputs.shape[1],15), 'param_grads': np.random.rand(inputs.shape[1],15)}])

model = Model()
model.learn(num_epochs,inputs,param_info,targets)


