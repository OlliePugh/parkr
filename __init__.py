import numpy as np
import json

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html

class Network:
    def __init__(self, input_nodes, hidden_layers, output_nodes, activation_function):
        self.layer_sizes = [input_nodes] + hidden_layers + [output_nodes]  # a list that holds the amount of nodes in each layer
        self.__generate_weights(self.layer_sizes)
        self.__generate_bias(self.layer_sizes)

    def __generate_weights(self, layer_sizes):
        self.weight_matrix = []

        for index, amount_of_nodes in enumerate(self.layer_sizes[:-1]):  # for all layers but output
            self.weight_matrix.append(np.random.randn(amount_of_nodes, self.layer_sizes[index+1]) * np.sqrt(2.0/amount_of_nodes))

    def __generate_bias(self, layer_sizes):
        self.bias_matrix = []

        for index, amount_of_nodes in enumerate(self.layer_sizes[1:]):  # for all layers but input as input nodes can not have bias
            self.bias_matrix.append(np.full((1, amount_of_nodes), 0.1))

    def print(self):
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


    def feed_forward(self, inp, return_entire_network=False):
        activated_node_value_matrix = [inp]
        node_value_matrix = [inp]
        
        for i in range(len(self.layer_sizes)-1):  # for each layer
            raw_out = np.dot(node_value_matrix[-1], self.weight_matrix[i])
            node_value_matrix.append(raw_out)
            activated_node_value_matrix.append(sigmoid(raw_out))

        if return_entire_network:
            return (node_value_matrix, activated_node_value_matrix)
        else:
            return (node_value_matrix[-1], activated_node_value_matrix[-1])

    def train(self, inp, expected, step_size=0.1):
        for i in range(len(inp)-1):
            raw_forward_pass, activated_forward_pass = self.feed_forward(inp[i], return_entire_network=True)
            self.__backprop(raw_forward_pass, activated_forward_pass, expected[i], step_size)

    def __backprop(self, raw_result, activated_result, expected, step_size):
        prev_layer_error = None
        prev_cost = None
        for i in (range(len(self.layer_sizes)-1, 0 ,-1)):  # for all layers but the input one
            if i == len(self.layer_sizes)-1:  # output layer
                curr_layer_error = (activated_result[-1] - expected) * derivative_sigmoid(activated_result[-1])             
            else:
                curr_layer_error = prev_layer_error * self.weight_matrix[i-1] * derivative_sigmoid(activated_result[i])

            curr_cost = curr_layer_error * raw_result[i]

            self.weight_matrix[i-1] -= curr_cost * step_size
            self.bias_matrix[i-1] += curr_layer_error * step_size  # FIXME Something wrong with the logic here

            prev_cost = curr_cost
            prev_layer_error = curr_layer_error

    def save(self, file_name):
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

def sigmoid(value):
    return 1.0 / (1 + np.exp(-value))

def derivative_sigmoid(value):
    return sigmoid(value) * (1-sigmoid(value))

def relu(value):
    return np.maximum(0, value)

def derivative_relu(value):
    return (value > 0).astype(int)

def cost(yHat, y):
    return 0.5 * (yHat - y)**2

def cost_prime(yHat, y):
    return yHat - y