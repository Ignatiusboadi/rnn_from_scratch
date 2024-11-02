import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size, l_rate=0.01):
        """

        :param input_size:
        :type input_size:
        :param hidden_size:
        :type hidden_size:
        :param output_size:
        :type output_size:
        :param l_rate:
        :type l_rate:
        """
        self.hidden_size = hidden_size
        self.l_rate = l_rate
        self.input_size = input_size
        self.output_size = output_size

        self.Wxh = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(self.output_size, self.hidden_size) * 0.01

        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.output_size, 1))
