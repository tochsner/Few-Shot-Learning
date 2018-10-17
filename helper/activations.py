import numpy as np

sigmoidActivation = lambda x: 1.0 / (1 + np.exp(-x))
sigmoidDerivation = lambda x: x * (1 - x)