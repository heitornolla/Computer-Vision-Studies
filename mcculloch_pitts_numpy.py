import numpy as np

# This code implements a eural Network based on the McCulloch-Pitts model from 1943

class NeuralNetwork():
  def __init__(self, inputs, weights, biases):
    self.inputs = inputs
    self.weights = weights
    self.biases = biases


  def __repr__(self):
    return f"{len(self.inputs)} inputs, {len(self.weights)} weights and {len(self.biases)} biases"
    

  def get_output(self):
    layer_outputs = np.dot(self.weights, self.inputs) + self.biases

    return layer_outputs


def main():
  inputs = [1, 2, 3, 2.5]

  ##LIST OF WEIGHTS
  weights = [[0.2, 0.8, -0.5, 1],
  [0.5, -0.91, 0.26, -0.5],
  [-0.26, -0.27, 0.17, 0.87]]

  ##LIST OF BIASES
  biases = [2, 3, 0.5]

  neural_network = NeuralNetwork(inputs, weights, biases)

  print(neural_network.get_output())


if __name__ == "__main__":
  main()