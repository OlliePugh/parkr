from .activation import Activation, ActivationMethods
from typing import List, Tuple

import numpy as np
import json
import math

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
            self.bias_matrix.append(np.random.randn(1, amount_of_nodes) * np.sqrt(2.0/amount_of_nodes))


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


    def forward_pass(self, inp: List[float], return_entire_network: bool=False) -> Tuple[List[List[float]], List[List[float]]]:
        """Perform a forward pass on the network

        Args:
            inp (List[List[double]]): The input values to the network
            return_entire_network (bool, optional): if true will return all of the network, not just the outputs. Defaults to False.

        Returns:
            (List[float], List[float]): A list of the raw values and the activated networks 
        """
        if len(inp) != self.layer_sizes[0]: raise ValueError(f"{len(inp)} inputs provided when the network has {self.layer_sizes[0]} input nodes")

        activated_node_value_matrix = [np.array(inp)]
        node_value_matrix = [np.array(inp)]
        
        for i in range(len(self.layer_sizes)-1):  # for each layer after the input layer
            raw_out = np.dot(activated_node_value_matrix[i], self.weight_matrix[i])
            raw_out = raw_out + self.bias_matrix[i]
            node_value_matrix.append(raw_out)
            activated_node_value_matrix.append(Activation.activate(self.activation_function, raw_out))

        if return_entire_network:
            return activated_node_value_matrix
        else:
            return activated_node_value_matrix[-1]

    def train(self, epochs:int, training_data:Tuple[List[List[float]],List[List[float]]], step_size:float=0.1, **kwargs) -> None:
        """Train the network

        Args:
            epochs (int): Amount of epochs to be performed
            training_data (List[List[float]]): Data to be trained on
            step_size (float, optional): Learning rate, higher will cause larger changes. Defaults to 0.1.
            batch_size (float, optional): The amount of rows in each batch. Disabled by default.
            export_name (string, optional): The file name to output the training loss and validation loss to as a csv. Disabled by default.
            momentum_rate (float, optional): The rate at which you want momentum to occur in the network. Disabled by default.
            bold_driver (Tuple[float, float, float, float], optional): The treshold to change the step size, the increase multiplier, and the decrease multiplier. And how frequently to check, in epochs. Disabled by default.
        """

        in_data, expected_data = training_data

        # check the inputs are valid

        if len(in_data[0]) != self.layer_sizes[0]: raise ValueError(f"{len(inp)} inputs provided when the network has {self.layer_sizes[0]} input nodes")
        
        if len(expected_data[0]) != self.layer_sizes[-1]: raise ValueError(f"{len(expected_data[0])} outputs provided when the network has {self.layer_sizes[-1]} output nodes")

        training_batches = [in_data]
        expected_batches = [expected_data]
        
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
            start_val = 0

            training_batches = []  # convert back to empty lists
            expected_batches = []
            for i in range(math.ceil(len(in_data)/batch_size)):
                ending_index = min(len(in_data)-1, start_val+batch_size)  # this will ensure that we dont go over the end of the array

                training_batches.append(in_data[start_val:ending_index])
                expected_batches.append(expected_data[start_val:ending_index])

                start_val += batch_size

        if "export_name" in kwargs:
            export_file = open(f"{kwargs['export_name']}.csv", "w+")
            export_file.write("Training Loss, Validation Loss\n")

        prev_weight_change = []
        prev_bias_change = []

        last_bd = (None , self.weight_matrix, self.bias_matrix)

        for epoch in range(epochs):  # for each epoch

            training_costs = []  # array to store training costs

            for batch_count, in_batch in enumerate(training_batches):

                # create templates for weight and bias changes

                avg_weight_change = []

                for layer in self.weight_matrix:
                    avg_weight_change.append(np.zeros(layer.shape))

                avg_bias_change = []

                for layer in self.bias_matrix:
                    avg_bias_change.append(np.zeros(layer.shape))

                # cycle through the batch of data

                for i in range(len(in_batch)):  # for each row in the batch
                    activated_forward_pass = self.forward_pass(in_batch[i], return_entire_network=True)  # get forward pass result
                    row_weight_change, row_bias_change = self.__generate_changes(activated_forward_pass, expected_batches[batch_count][i], step_size, prev_changes=(prev_weight_change, prev_bias_change), **kwargs)  # generate the changes required

                    for index, layer in enumerate(row_weight_change):
                        avg_weight_change[index] = avg_weight_change[index] + (layer/len(in_batch))
                    for index, layer in enumerate(row_bias_change):
                        avg_bias_change[index] = avg_bias_change[index] + (layer.reshape((1,-1))/len(in_batch))
        
                    this_pass_cost = cost(activated_forward_pass[-1], expected_batches[batch_count][i], len(expected_data[0]))

                    training_costs.append(np.mean(this_pass_cost))  # get mean of all output nodes cost

                self.__backprop(avg_weight_change, avg_bias_change)  

                prev_weight_change = avg_weight_change
                prev_bias_change = avg_bias_change

            training_loss = round(np.mean(training_costs), 7)
            to_print = f"Epoch {epoch+1}: TLoss: {training_loss}"

            if "validation" in kwargs:
                validation_data, validation_expected_data = kwargs["validation"] 
                validation_costs = []
                for index, row in enumerate(validation_data):
                    result = self.forward_pass(row)
                    validation_costs.append(cost(result, validation_expected_data[index], len(validation_expected_data[0])))

                validation_loss = round(np.mean(validation_costs), 7)
                to_print += f" VLoss: {validation_loss}"

            # check bold driver
            if ("bold_driver" in kwargs and epoch % kwargs["bold_driver"][3] == 0):
                bd_threshold, bd_increase, bd_decrease, bd_frequency = kwargs["bold_driver"]
                if (epoch != 0):  # it is not the first epoch therefore can affect step_size
                    difference = (validation_loss - last_bd[0])/max(last_bd[0], validation_loss)
                    
                    if difference > bd_threshold:  # decrease step_size
                        step_size = max(step_size * bd_decrease, 0.01)
                        self.weight_matrix = last_bd[1]  # reset weights and bias' to before for better training
                        self.bias_matrix = last_bd[2]

                    elif difference < bd_threshold*-1:  # increase step_size
                        step_size = min(step_size * bd_increase, 0.5)

                last_bd = (validation_loss, self.weight_matrix, self.bias_matrix)


            if "export_name" in kwargs:  # output the loss on that back prop to the output file
                export_file.write(f"{training_loss}, {validation_loss}\n")

            print(to_print)

        if "export_name" in kwargs:
            export_file.close()  # close the export file

    def __generate_changes(self, activated_result: List[List[float]],  expected: List[float], step_size: float, prev_changes, **kwargs) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate changes for the weights and bias' from a forward pass and the expected results

        Args:
            activated_result (List[List[float]]): The result of the forwad pass with the activation function applied
            expected (List[float]): The expected values of the network
            step_size (float): The learning rate of the network

        Returns:
            Tuple[List[List[float]], List[List[float]]]: First value is weight changes, seconc value is bias changes
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

            new_weight = step_size * (np.dot(activated_result[i].reshape(-1,1), delta_map[i].reshape(1,-1)))

            if "momentum_rate" in kwargs and prev_changes[0]:
                new_weight += kwargs["momentum_rate"] * prev_changes[0][i]  # add the values of the previous weight changes of the layer

            weight_changes.append(new_weight)

        for i in range(len(self.bias_matrix)):
            #update bias

            new_bias = step_size * delta_map[i]

            if "momentum_rate" in kwargs and prev_changes[1]:
                new_bias += kwargs["momentum_rate"] * prev_changes[1][i].T  # add the values of the previous bias changes of the layer

            bias_changes.append(new_bias)

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
            self.bias_matrix[i] = self.bias_matrix[i] + bias_change[i]


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


def cost(predicted, actual, rows_of_data):
    return ((predicted - actual)**2) / rows_of_data