from .activation import Activation, ActivationMethods

import numpy as np
import json

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class Network:
    def __init__(self, input_nodes, hidden_layers, output_nodes, activation_function):
        if (not isinstance(activation_function, ActivationMethods)):
            raise ValueError("Unkown Activation Method")
        
        self.activation_function = activation_function
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

    def train(self, inp, expected, step_size=0.1):
        for i in range(len(inp)):
            raw_forward_pass, activated_forward_pass = self.feed_forward(inp[i], return_entire_network=True)
            self.__backprop(raw_forward_pass, activated_forward_pass, expected[i], step_size)

    def __backprop(self, raw_result, activated_result, expected, step_size): 
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
        
        for i in range(len(self.layer_sizes)-1):  # for each layer from the first layer
            
            #  update the weights
            weight_change = (step_size * (np.dot(activated_result[i].reshape(-1,1), delta_map[i].reshape(1,-1))))
            self.weight_matrix[i] = self.weight_matrix[i] + weight_change.reshape(self.weight_matrix[i].shape)
            
        for i in range(len(self.bias_matrix)):
            bias_change = (step_size * delta_map[i])
            self.bias_matrix[i] = self.bias_matrix[i] + bias_change.reshape((1,-1))


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

def cost(yHat, y):
    return 0.5 * (yHat - y)**2

def cost_prime(yHat, y):
    return yHat - y