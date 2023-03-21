import numpy as np
from numpy import ndarray
from linearRegression import LinearRegression

learning_rate = 0.01
num_epochs = 100
param_info = dict({'linReg1',[1,2,3,4]},{'biasAdd1',{1,1,1,1}})
linearRegressionOp = LinearRegression();

class Model(object):
    def __init__(self) -> None:
        pass

    def main(self):
        for i in range(num_epochs):
            output = self.network_forward()
            gradients = self.network_backward()



    def network_forward(self):



    def network_backward(self):