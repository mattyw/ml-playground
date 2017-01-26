import unittest
import numpy as np
from nn import NeuralNetwork


class NNTest(unittest.TestCase):

    def test_simple(self):
        nn = NeuralNetwork(3, 3, 3, 0.3)
        nn.weights_input_hidden = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5
        ]], ndmin=2).T
        nn.weights_hidden_output = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5
        ]], ndmin=2).T
        result = nn.query([1.0, 0.5, -1.5])
        expected = np.array([0.6791787, 0.6791787, 0.6791787], ndmin=2).T
        self.assertEqual(str(result), str(expected)) # TODO: Comparing strings is terrible

if __name__ == '__main__':
    unittest.main()
