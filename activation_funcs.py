import numpy as np

class ReLu:
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)