from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

class ActivationMethods(Enum):
    SIGMOID=1
    RELU=2

class Activation(ABC):

    @abstractmethod
    def activate(method, value):
        
        if method == ActivationMethods.SIGMOID:
            return sigmoid(value)

        elif method == ActivationMethods.RELU:
            return (relu(value))

        raise ValueError ("Unkown method specified")

    @abstractmethod
    def derivative_activate(method, value):
        
        if method == ActivationMethods.SIGMOID:
            return derivative_sigmoid(value)

        elif method == ActivationMethods.RELU:
            return (derivative_relu(value))

        raise ValueError ("Unkown method specified")

def sigmoid(value):
    return 1.0 / (1 + np.exp(-value))

def derivative_sigmoid(value):
    return value * (1-value)

def relu(value):
    return np.maximum(0, value)

def derivative_relu(value):
    return (value > 0).astype(int)