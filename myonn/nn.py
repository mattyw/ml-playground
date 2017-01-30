import numpy as np
import scipy.special as spspec


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.weights_input_hidden = (np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        self.weights_hidden_output = (np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)

        self.activation_function = spspec.expit

    def train(self, input_list, target_list):
        # Convert to matrix
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden = self._traverse_layer(inputs, self.weights_input_hidden)
        final_outputs = self._traverse_layer(hidden, self.weights_hidden_output)

        # Find the errors
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        # Update weights for hidden -> output layers
        self.weights_hidden_output += self._update_weight(hidden_outputs, final_outputs, output_errors)

        # Update weights for inputs -> output hidden
        self.weights_input_hidden += self._update_weight(hidden_errors, hidden_outputs, inputs)

    def _update_weight(self, inputs, outputs, errors):
        """ Given some inputs -> outputs and the error for each return the adjustment to be made to
        the weights
        """
        return self.learning_rate * np.dot((errors, outputs * (1.0 - outputs)), np.transpose(inputs))

    def _traverse_layer(self, inputs, input_weights):
        a_input = np.dot(input_weights, inputs)
        return self.activation_function(a_input)

    def query(self, input_list):
        # convert inputs to 2d array (matrix)
        inputs = np.array(input_list, ndmin=2).T

        hidden = self._traverse_layer(inputs, self.weights_input_hidden)
        return self._traverse_layer(hidden, self.weights_hidden_output)

if __name__ == '__main__':
    nn = NeuralNetwork(3, 3, 3, 0.3)
    input = [1.0, 0.5, -1.5]
    print(nn.query(input))
    target = [0.5, 1.0, 1.0]
    for x in range(1000):
        nn.train(input, target)
    print(nn.query(input))
