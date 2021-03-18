from .activation import Activation, ActivationMethods
from typing import List, Tuple

import numpy as np
import json

class Network:
    def __init__(self, input_nodes: int, hidden_layers: List[int], output_nodes: int, activation_function: ActivationMethods):
        """
        Args:
            input_nodes (int): Amount of input nodes in network
            hidden_layers (List[int]): List of amount of nodes in each elements
            output_nodes (int): Amount of output nodes in network
            activation_function (ActivationMethods): Activation function to be used

        Raises:
            ValueError: Raised if the activation method is a valid ActivationMethod value
        """

        if (not isinstance(activation_function, ActivationMethods)):
            raise ValueError("Unkown Activation Method")
        
        self.activation_function = activation_function
        self.layer_sizes = [input_nodes] + hidden_layers + [output_nodes]  # a list that holds the amount of nodes in each layer
        self.__generate_weights(self.layer_sizes)
        self.__generate_bias(self.layer_sizes)

    def __generate_weights(self, layer_sizes: List[int]) -> None:
        """Generate the weights for the network

        Args:
            layer_sizes (List[int]): The amounts of nodes in each layers
        """
        self.weight_matrix = []

        for index, amount_of_nodes in enumerate(self.layer_sizes[:-1]):  # for all layers but output
            self.weight_matrix.append(np.random.randn(amount_of_nodes, self.layer_sizes[index+1]) * np.sqrt(2.0/amount_of_nodes))

    def __generate_bias(self, layer_sizes: List[int]) -> None:
        """Generate the bias' for the network

        Args:
            layer_sizes (List[int]): The amounts of nodes in each layers
        """
        self.bias_matrix = []

        for index, amount_of_nodes in enumerate(self.layer_sizes[1:]):  # for all layers but input as input nodes can not have bias
            self.bias_matrix.append(np.full((1, amount_of_nodes), 0.1))

    def print(self) -> None:
        """Print the structure of the network to the console
        """
        curr_char = 65
        for i, node_count in enumerate(self.layer_sizes):
            for j in range(node_count):
                if i == 0:
                    print(f"{chr(curr_char)}(0) ", end="")
                else:
                    print(f"{chr(curr_char)}({round(self.bias_matrix[i-1][0][j], 5)}) ", end="") 
                curr_char += 1
            print("\n")

        # print the weights values

        node_offset = 0
        for cols in self.weight_matrix:  # for each gap
            for i, rows in enumerate(cols):
                for j, weight in enumerate(rows):
                    pass
                    print(f"{chr(65+node_offset+i)}->{chr(65+node_offset+j+len(cols))}: {round(weight,5)}")
            node_offset += len(cols)


    def feed_forward(self, inp: List[List[float]], return_entire_network: bool=False) -> Tuple[List[List[float]], List[List[float]]]:
        """Perform a forward pass on the network

        Args:
            inp (List[List[double]]): The input values to the network
            return_entire_network (bool, optional): if true will return all of the network, not just the outputs. Defaults to False.

        Returns:
            (List[float], List[float]): A list of the raw values and the activated networks 
        """
        activated_node_value_matrix = [np.array(inp)]
        node_value_matrix = [np.array(inp)]
        
        for i in range(len(self.layer_sizes)-1):  # for each layer after the input layer
            raw_out = np.dot(activated_node_value_matrix[i], self.weight_matrix[i])
            raw_out = raw_out + self.bias_matrix[i]
            node_value_matrix.append(raw_out)
            activated_node_value_matrix.append(Activation.activate(self.activation_function, raw_out))

        if return_entire_network:
            return (node_value_matrix, activated_node_value_matrix)
        else:
            return (node_value_matrix[-1], activated_node_value_matrix[-1])

    def train(self, epochs:int, training_data:List[List[float]], step_size:float=0.1, **kwargs) -> None:
        """Train the network

        Args:
            epochs (int): Amount of epochs to be performed
            training_data (List[List[float]]): Data to be trained on
            step_size (float, optional): Learning rate, higher will cause larger changes. Defaults to 0.1.
        """

        for epoch in range(epochs):
            costs = []
            weight_changes = []
            bias_changes = []
            for i in range(len(training_data[0])):
                raw_forward_pass, activated_forward_pass = self.feed_forward(training_data[0][i], return_entire_network=True)
                
                row_weight_change, row_bias_change = self.__generate_changes(activated_forward_pass, training_data[1][i], step_size)
                
                weight_changes.append(row_weight_change)
                bias_changes.append(row_bias_change)
                costs.append(np.mean(cost(activated_forward_pass[-1], training_data[1])))

            print(f"Epoch {epoch}: {round(np.mean(costs),7)}")

            #FIXME only training for last row currently 

            # get average weight and bias changes
            avg_weight_change = []
            for layer in weight_changes[0]:
                avg_weight_change.append(np.zeros(layer.shape))

            for weight_change in weight_changes:
                for index, layer in enumerate(weight_change):
                    avg_weight_change[index] += + layer/len(weight_changes)

            avg_bias_change = []
            for layer in bias_changes[0]:
                avg_bias_change.append(np.zeros(layer.shape))

            for bias_change in bias_changes:
                for index, layer in enumerate(bias_change):
                    avg_bias_change[index] += layer/len(bias_changes)

            self.__backprop(avg_weight_change, avg_bias_change)  
            

    def __generate_changes(self, activated_result: List[List[float]],  expected: List[float], step_size: float) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate changes for the weights and bias' from a forward pass and the expected results

        Args:
            activated_result (List[List[float]]): The result of the forwad pass with the activation function applied
            expected (List[float]): The expected values of the network
            step_size (float): The learning rate of the network

        Returns:
            Tuple[List[List[float]], List[List[float]]]: [description]
        """
        
        # create delta map
        delta_map = []
        for i in (range(len(self.layer_sizes)-1, 0 ,-1)):  # for all layers but the input one
            if i == len(self.layer_sizes)-1:  # output layer
                derived = Activation.derivative_activate(self.activation_function, activated_result[i])
                delta = np.dot((expected-activated_result[i]), derived)
                delta_map.insert(0,delta)
            else:
                delta = np.dot(self.weight_matrix[i], delta_map[0].reshape((-1,1)))
                deriv_sigmoid = Activation.derivative_activate(self.activation_function, activated_result[i].reshape((-1,1)))
                delta_map.insert(0,(delta * deriv_sigmoid))

        weight_changes = []
        bias_changes = []

        for i in range(len(self.layer_sizes)-1):  # for each layer from the first layer
            #  update the weights
            weight_changes.append(step_size * (np.dot(activated_result[i].reshape(-1,1), delta_map[i].reshape(1,-1))))

        for i in range(len(self.bias_matrix)):
            #update bias
            bias_changes.append(step_size * delta_map[i])

        return (weight_changes, bias_changes)

    def __backprop(self, weight_change: List[List[float]], bias_change: List[List[float]]) -> None: 
        """Update the weights and bias' inside the network

        Args:
            weight_change (List[List[float]]): A list of changes to be added to the weights
            bias_change (List[List[float]]): A list of changes to be added to the bias
        """
        
        for i in range(len(self.layer_sizes)-1):  # for each layer from the first layer
            #  update the weights
            self.weight_matrix[i] = self.weight_matrix[i] + weight_change[i].reshape(self.weight_matrix[i].shape)
            
        for i in range(len(self.bias_matrix)):
            self.bias_matrix[i] = self.bias_matrix[i] + bias_change[i].reshape((1,-1))


    def save(self, file_name: str) -> None:   # TODO Allow for loading from disk
        """Save the network to disk

        Args:
            file_name (str): name of the file
        """
        with open(f"{file_name}.json", "w+") as file:
            file.write(json.dumps(self.__dict__, default=self.__json_parser, indent=2))
        

    @staticmethod
    def __json_parser(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))

def cost(predicted, actual):
    return 0.5 * (predicted - actual)**2

def cost_prime(predicted, actual):
    return predicted - actual