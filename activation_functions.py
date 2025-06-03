import numpy as np

class ReLu:
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)

class Softmax:
  def forward(self, inputs):
    exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exponents / np.sum(exponents, axis=1, keepdims=True)

    self.output = probabilities