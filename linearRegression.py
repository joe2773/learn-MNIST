import numpy as np
from numpy import ndarray

class LinearRegression(object):
    def __init__(self) -> None:
        pass

    def _forward(self,inputs :ndarray, params: ndarray) -> ndarray:
        self.inputs = inputs
        self.params = params
        return np.dot(inputs,params)

    def backward(self,output_grads: ndarray) -> ndarray:
        return np.transpose(self.params)

    def param_grad(self) -> ndarray:
        return np.transpose(self.inputs)