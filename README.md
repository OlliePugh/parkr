# Parker 
## A light weight neural network library

Parker was created for my second year AI methods module at university.

It is very basic to get a neural network up and running and getting results quickly.

## Examples

### Creating the network

To create a network you just need to specify how many input nodes, how many output nodes, a vector that contains the amount of nodes in each hidden layer, and the activation function you want to use.

```
Network myNetwork = Network(6, 2, {4,4}, Activation::TANH);
```

This will create a network with 6 input nodes, 2 hidden layers, both with 4 nodes each and 2 output nodes and it will use the tanh activation function.

### Training the network

To train the network you pass two datasets (which are just a typedef for `vector<vector<double>>`). One which contains the training data and one which contains the expected values.

Then when calling the train method the network will updates its bias' and weights on those values.

```
dataset trainingData = {{1,0},
                        {0,1}};

dataset expectedData = {{1},{0}};

int epochs = 10;
myNetwork.train(epochs, &trainingData, &expectedData);
```

### Performing a forward pass on the network

To perform a forward pass on the network you simply pass it a vector with each element as an input vaule.

```
vector<double> myResult = myNetwork.forwardPass({1,0},true);
```

If the last parameter in `forwardPass` is true the value of the forward pass will be printed, this is false by default. 

### Saving and opening networks

Parker supports saving networks to disk in the format of a .prkr file. You can do this as follows

#### Save
```
myNetwork.save("my_network_on_disk");
```

#### Load
```
Network mySavedNetwork = Network::open("my_network_on_disk");
```
