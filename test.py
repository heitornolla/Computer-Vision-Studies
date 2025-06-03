import numpy as np

from nnfs.datasets import spiral_data

import numpy as np

from src.activation_functions import ReLu, Softmax
from src.dense_layer import DenseLayer


def generate_data():
  x, y = spiral_data(100, 3)
  return x,y


def get_neural_network(inputs):
  dense1 = DenseLayer(2, 3)
  activation1 = ReLu()
  dense2 = DenseLayer(3, 3)
  activation2 = Softmax()

  dense1.forward(inputs)
  activation1.forward(dense1.output)
  dense2.forward(activation1.output)
  activation2.forward(dense2.output)

  return activation2.output


def main():
  x, y = generate_data()

  result = get_neural_network(x)

  print(result[:5])


if __name__ == "__main__":
  main()