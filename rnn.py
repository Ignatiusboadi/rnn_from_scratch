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
        self.outputs = None
        self.inputs = None
        self.hidden_states = None
        self.hidden_size = hidden_size
        self.l_rate = l_rate
        self.input_size = input_size
        self.output_size = output_size

        self.Wxh = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(self.output_size, self.hidden_size) * 0.01

        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.output_size, 1))

    def forward(self, inputs):
        hidden_state = np.zeros((self.hidden_size, 1))
        self.hidden_states = [hidden_state]
        self.inputs = inputs
        self.outputs = []
        for x in self.inputs:
            x = x.reshape(-1, 1)
            hidden_state = np.tanh(self.Wxh @ x + self.Whh @ hidden_state + self.bh)
            self.hidden_states.append(hidden_state)
            output = self.Why @ hidden_state + self.by
            self.outputs.append(output)

        return self.outputs, self.hidden_states[-1]
