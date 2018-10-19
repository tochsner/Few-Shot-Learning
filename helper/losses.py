import numpy as np

class MeanSquareCostFunction():
    def get_cost(self, output_values, correct_values):
        return np.sum(np.power(output_values - correct_values, 2))
    def get_derivatives(self, output_values, correct_values):
        return outputValues - correctValues