# Parkr 
## A light weight neural network library

[![Python package](https://github.com/OlliePugh/parkr/actions/workflows/python-package.yml/badge.svg?event=push)](https://github.com/OlliePugh/parkr/actions/workflows/python-package.yml)

Parkr was created for my second year AI methods module at university.

It is very basic to get a neural network up and running and getting results quickly.

## Installation

```
pip install parkr
```

## Examples

### Creating the network

To create a network you just need to specify how many input nodes, how many output nodes, a vector that contains the amount of nodes in each hidden layer, and the activation function you want to use.

```
from parkr import Network, ActivationMethods
my_network = Network(6, [4,4], 2, ActivationMethods.TANH)
```

This will create a network with 6 input nodes, 2 hidden layers, both with 4 nodes each and 2 output nodes and it will use the tanh activation function.

### Training the network

To train the network you must pass a tuple where the first element is the training data and the second is the expected values for that training data.

Then when calling the train method the network will updates its bias' and weights on those values.

```
import numpy as np
training_data = np.array(
    [0.77,0.21,0.34,0.2,0.12,0.54],
    [0.43,0.12,0.65,0.12,0.76,0.34]
    )

expected_data = np.array(
    [1],
    [0]
    )

epochs = 10
my_network.train(epochs, (training_data, expected_data))
```

### Performing a forward pass on the network

To perform a forward pass on the network you simply pass it a vector with each element as an input vaule.

```
my_result = my_network.feed_forward([[0.54,0.32,0.12,0.46,0.68,0.43]])
```

### Saving and opening networks

Parkr supports saving networks to disk and loading them back in so that you do not have to retrain your network everytime it leaves memory.
#### Save
This will save the networks structure as a .json file 
```
my_network.save("my_network_on_disk")
```

#### Load
This will load the network from the .json file
```
my_loaded_network = Network.open("my_network_on_disk")
```

### Print
This will output the structure of the network along with the weights and bias' to the console.
```
my_network = Network(4,[6,2],2, ActivationMethods.SIGMOID)
my_network.print()
```
Would output the following
```
A(0) B(0) C(0) D(0) 

E(0.30882) F(-0.27833) G(0.74767) H(-0.20595) I(-0.54466) J(0.32999)

K(-1.64158) L(0.17655)

M(-0.02446) N(-0.23844)

A->E: 1.65803
A->F: 0.88988
A->G: -0.20031
A->H: 0.0684
A->I: 0.50825
A->J: 0.69463
B->E: 0.11407
B->F: 0.30879
B->G: -0.06734
B->H: 0.45021
```
