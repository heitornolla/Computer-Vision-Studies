import numpy as np

from nnfs.datasets import spiral_data

import numpy as np


def generate_data():
  x, y = spiral_data(100, 3)
  return x,y


class DenseLayer():
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases


def main():
  x, y = generate_data()

  dense1 = DenseLayer(2, 3)
  dense1.forward(x)

  print(dense1.output[:5])


if __name__ == "__main__":
  main()