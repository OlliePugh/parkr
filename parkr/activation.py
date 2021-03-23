from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

class ActivationMethods(Enum):
    SIGMOID=1
    RELU=2
    TANH=3
    LEAKY_RELU=4
    LINEAR=5

class Activation(ABC):

    @abstractmethod
    def activate(method: ActivationMethods, value: float) -> float:
        """Perform activation method on a value

        Args:
            method (ActivationMethods): [description]
            value (float): The value to be altered

        Raises:
            ValueError: Raised if unknown activation method specifed

        Returns:
            float: Value with the activation function applied
        """
        
        if method == ActivationMethods.SIGMOID:
            return sigmoid(value)

        elif method == ActivationMethods.RELU:
            return relu(value)
        
        elif method == ActivationMethods.TANH:
            return np.tanh(value)

        elif method == ActivationMethods.LEAKY_RELU:
            return leaky_relu(value)
        elif method == ActivationMethods.LINEAR:
            return value

        raise ValueError ("Unkown method specified")

    @abstractmethod
    def derivative_activate(method: ActivationMethods, value: float) -> float:
        """Perform derivative activation method on a value

        Args:
            method (ActivationMethods): [description]
            value (float): The value to be altered

        Raises:
            ValueError: Raised if unknown activation method specifed

        Returns:
            float: Value with the derivative activation function applied
        """
        
        if method == ActivationMethods.SIGMOID:
            return derivative_sigmoid(value)

        elif method == ActivationMethods.RELU:
            return derivative_relu(value)

        elif method == ActivationMethods.TANH:
            return 1-pow(np.tanh(value), 2.0)

        elif method == ActivationMethods.LEAKY_RELU:
            return derivative_leaky_relu(value)
        
        elif method == ActivationMethods.LINEAR:
            return 1

        raise ValueError ("Unkown method specified")

def sigmoid(value):
    return 1.0 / (1 + np.exp(-value))

def derivative_sigmoid(value):
    return value * (1-value)

def relu(value):
    return np.maximum(0, value)

def derivative_relu(value):
    return (value > 0).astype(int)

def leaky_relu(value, alpha=0.01):
    return np.where(value > 0, value, alpha * value) 

def derivative_leaky_relu(value, alpha=0.01):
    return np.where(value > 0, 1, alpha)