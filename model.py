import numpy as np
from linearRegression import LinearRegression
from mean_squared_error_loss import MSELoss
from numpy import ndarray

class Model(object):
    def __init__(self) -> None:
        self.linearRegressionOp = LinearRegression()
        self.lossOp = MSELoss()

    def learn(self, num_epochs:int,inputs: ndarray,param_info :ndarray,targets: ndarray):
        self.param_info = param_info
        for i in range(num_epochs):
            outputs = self.network_forward(inputs,param_info,targets)
            loss = self.lossOp._forward(outputs,targets)
            loss_grad = self.lossOp._backward(inputs,targets)
            self.param_info = self.network_backward(loss_grad)
            print(loss)
            

    def network_forward(self, inputs: ndarray,param_info :ndarray,targets:ndarray) -> ndarray:
        outputs = self.linearRegressionOp._forward(inputs, param_info[0]['params'])
        return outputs

    def network_backward(self,loss_grad: ndarray) -> ndarray:
        output_grad = self.linearRegressionOp.backward(loss_grad)
        self.param_info[0]['param_grads'] = self.linearRegressionOp.param_grad()
        return self.param_info
    