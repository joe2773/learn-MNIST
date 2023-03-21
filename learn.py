import numpy as np
from numpy import ndarray
from linearRegression import LinearRegression

learning_rate = 0.01
num_epochs = 100
param_info = dict({'linReg1',[1,2,3,4]},{'biasAdd1',{1,1,1,1}})
input_data = []
linearRegressionOp = LinearRegression();

class Model(object):
    def __init__(self) -> None:
        pass

    def main(self):
        for i in range(num_epochs):
            outputs = self.network_forward()
            loss = self.lossOp.loss()
            loss_grad = self.lossOp.loss_grad()
            param_gradients = self.network_backward(loss_grad)
            self.param_info = self.optimiser.optimise(param_gradients,self.param_info)


    def network_forward(self, inputs: ndarray):
        output = linearRegressionOp._forward(inputs)
        return output

    def network_backward(self,loss_grad: ndarray):
        output_grad = linearRegressionOp.backward(loss_grad)
        self.param_info['linReg1Grads'] = linearRegressionOp.param_grad()
        return self.param_info
    