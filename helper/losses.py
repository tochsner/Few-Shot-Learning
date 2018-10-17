class MeanSquareCostFunction():
    def getCost(self, outputValues, correctValues):
        return np.sum(np.power(outputValues - correctValues, 2))
    def getIndividualCost(self, outputValues, correctValues):
        return np.power(outputValues - correctValues, 2)    
    def getDerivatives(self, outputValues, correctValues):
        return outputValues - correctValues