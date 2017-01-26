import numpy as np
import scipy.special as spspec


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.lr = learning_rate

        self.weights_input_hidden = (np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        self.weights_hidden_output = (np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)

        self.activation_function = spspec.expit

    def train(self, input_list, target_list):
        # Convert to matrix
        inputs = np.array(input_list, ndmin=2).T
        target = np.array(target_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = np.dot(self.weights_hidden_output, hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_output = self.activation_function(final_inputs)

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
    print(nn.query([1.0, 0.5, -1.5]))
