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

        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = np.dot(self.weights_hidden_output, hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Find the errors
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        # Update weights for hidden -> output layers
        self.weights_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        # Update weights for inputs -> output hidden
        self.weights_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, input_list):
        # convert inputs to 2d array (matrix)
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

if __name__ == '__main__':
    nn = NeuralNetwork(3, 3, 3, 0.3)
    input = [1.0, 0.5, -1.5]
    print(nn.query(input))
    target = [0.5, 1.0, 1.0]
    for x in range(1000):
        nn.train(input, target)
    print(nn.query(input))
