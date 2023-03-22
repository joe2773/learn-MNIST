import numpy as np
from numpy import ndarray

class MSELoss():
    def __init__(self) -> None:
        pass

    def _forward(self, inputs: ndarray,targets: ndarray) -> ndarray:
        return  (targets - inputs)**2/inputs.shape[0]

    def _backward(self, inputs: ndarray, targets: ndarray) -> ndarray:
        return 2*(targets - inputs)**2/inputs.shape[0]