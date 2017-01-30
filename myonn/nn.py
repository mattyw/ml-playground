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

    def _train(self, input_list, target_list):
        # Convert to matrix
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden = self._traverse_layer(inputs, self.weights_input_hidden)
        final_outputs = self._traverse_layer(hidden, self.weights_hidden_output)

        # Find the errors
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        # Update weights for hidden -> output layers
        self.weights_hidden_output += self._update_weight(hidden, final_outputs, output_errors)

        # Update weights for inputs -> output hidden
        self.weights_input_hidden += self._update_weight(hidden_errors, hidden_outputs, inputs)

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.weights_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.weights_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

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


def read_file(name):
    with open(name, 'r') as f:
        return f.readlines()


def train_mnist(nn, filename):
    data = read_file(filename)
    values = []
    for record in data:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.00) * 0.01
        targets = np.zeros(nn.output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        nn.train(inputs, targets)


# Returns (expected output, input)
def read_test(filename):
    data = read_file(filename)
    values = data[0].split(',')
    return values[0], (np.asfarray(values[1:]) / 255.0 * 0.00) * 0.01


def mnist():
    nn = NeuralNetwork(784, 100, 10, 0.3)
    train_mnist(nn, "mnist_train_100.csv")
    expected, input = read_test("mnist_test_10.csv")
    print("expected: ", expected)
    print(nn.query(input))


def test_main():
    nn = NeuralNetwork(3, 3, 3, 0.3)
    input = [1.0, 0.5, -1.5]
    print(nn.query(input))
    target = [0.5, 1.0, 1.0]
    for x in range(1000):
        nn.train(input, target)
    print(nn.query(input))

if __name__ == '__main__':
    mnist()
