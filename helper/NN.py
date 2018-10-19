import numpy as np  
import math       
import itertools   
import io         
import scipy as sp     
from scipy import stats                                                                                           


class BaseNetwork:
    def __init__(self):
        return None    
    def getOutput(self, inputValues):
        return None    
    def getDerivatives(self, outputDerivatives):
        return None
    def trainNetwork(self, inputValues, correctOutputValues):
        return None
    def applyChanges(self, learningRate, biasLearningRate, regularization):
        return None
class SimpleNeuronalNetwork(BaseNetwork):
    def __init__(self, size, activationFunction, activationDerivative, costFunction):
        BaseNetwork.__init__(self)

        if len(size) < 2 or min(size) == 0:
            raise ValueError("Size of network is not valid.")            
        
        self.InputNeuronsCount = size[0]
        self.OutputNeuronsCount = size[-1]
        self.Size = size
        
        self.ActivationFunction = activationFunction        
        self.ActivationDerivative = activationDerivative

        self.CostFunction = costFunction

        #Initialize network
        maxLayerSize = max(size)
        layerCount = len(size)
        self.LayerCount = layerCount

        self.Neurons = [np.zeros((x)) for x in size[:]]
        self.Bias = [np.zeros((x)) for x in size[:]]
        self.Weights = [np.random.randn(size[x],size[x+1]) for x in range(0, len(size)-1)]                                 

        self.NewWeights = [np.zeros((size[x],size[x+1])) for x in range(0, len(size)-1)]
        self.NewBias = [np.zeros((x)) for x in size[:]]
        self.BatchSize = 0

        self.Derivatives = [np.zeros((x)) for x in size[:]]        

    def load(self, path):
        #Load weights and bias from the files in 'path'
        BaseNetwork.__init__(self)

        self.Size = np.load(path + "/size.npy")
        self.Weights = np.load(path + "/weights.npy")
        self.Bias = np.load(path + "/bias.npy")        
        
        self.InputNeuronsCount = self.Size[0]
        self.OutputNeuronsCount = self.Size[-1]      

        print(self.Size) 

        #Initialize network        
        layerCount = len(self.Size)
        self.LayerCount = layerCount

        self.NewWeights = [np.zeros((self.Size[x],self.Size[x+1])) for x in range(0, len(self.Size)-1)]
        self.NewBias = [np.zeros((x)) for x in self.Size[:]]
        self.Batchsize = 0

        self.Derivatives = [np.zeros((x)) for x in self.Size[:]] 
                 
    def getOutput(self, inputValues):        
        self.Neurons[0] = inputValues 

        for i in range(self.LayerCount - 1):                                                           
            self.Neurons[i+1] = self.ActivationFunction(self.Weights[i].T.dot(self.Neurons[i]) + self.Bias[i+1])                        

        return self.Neurons[-1]
    
    def getDerivatives(self, outputDerivatives):
        #Backpropagates through the network, but doesn't change the weights
        self.Derivatives[-1] = self.ActivationDerivative(self.Neurons[-1]) * outputDerivatives                                                      
        
        #Change bias of output
        self.NewBias[-1] += self.Derivatives[-1]         
       
        for i in range(self.LayerCount - 2, 0, -1):    
            #Change weights                                               
            self.NewWeights[i] += (self.Neurons[i][np.newaxis].T * self.Derivatives[i+1])                                                                                          
            #Collect derivatives            
            self.Derivatives[i] = self.Weights[i].dot(self.Derivatives[i+1])

            #Compute derivative in respect to the input of the Neuron            
            self.Derivatives[i] *= self.ActivationDerivative(self.Neurons[i])
            #Change bias
            self.NewBias[i] += self.Derivatives[i]                                    
            
        #Change last weights                                               
        self.NewWeights[0] += (self.Neurons[0][np.newaxis].T * self.Derivatives[1])                                                                                          
        #Collect last derivatives            
        self.Derivatives[0] = self.Weights[0].dot(self.Derivatives[1])

        self.BatchSize += 1              
        return self.Derivatives[0]
    
    def getDerivativesWithoutChange(self, outputDerivatives):
        self.Derivatives[-1] = self.ActivationDerivative(self.Neurons[-1]) * outputDerivatives                                                                         
        for i in range(self.LayerCount - 2, 0, -1):    
            #Collect derivatives            
            self.Derivatives[i] = self.Weights[i].dot(self.Derivatives[i+1])

            #Compute derivative in respect to the input of the Neuron            
            self.Derivatives[i] *= self.ActivationDerivative(self.Neurons[i])
            
        #Collect last derivatives            
        self.Derivatives[0] = self.Weights[0].dot(self.Derivatives[1])

        return self.Derivatives[0]
    
    def trainNetwork(self, inputValues, correctOutputValues):
        #feed forward
        self.getOutput(inputValues)

        # return gradient
        return self.getDerivatives(self.CostFunction.get_derivatives(self.Neurons[-1], correctOutputValues))
        
    def applyChanges(self, learningRate, biasLearningRate, regularization, constrainWeights = False): 
        #Changes the weights after a batch
        self.Weights = [(w - nw * (learningRate / self.BatchSize) - regularization * w) for w, nw in zip(self.Weights, self.NewWeights)]
        self.Bias = [(b - nb * (biasLearningRate / self.BatchSize)) for b, nb in zip(self.Bias, self.NewBias)]                                                        

        self.NewWeights = [np.zeros(x.shape) for x in self.NewWeights]
        self.NewBias = [np.zeros(x.shape) for x in self.NewBias]                   

        self.BatchSize = 0            

    def evaluate(self, test_data):               
        test_results = [(np.argmax(self.getOutput(x[:,0])), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, path):
        #saves the weights and bias in 'path'
        np.save(path + "/weights", self.Weights)
        np.save(path + "/bias", self.Bias)
        np.save(path + "/size", self.Size)        



class HebbianNetwork(SimpleNeuronalNetwork): 
    def __init__(self, size, activationFunction):
        SimpleNeuronalNetwork.__init__(self, size, activationFunction, activationFunction, activationFunction)

    def getDerivatives(self, outputDerivatives):
        return outputDerivatives
        
    def trainNetwork(self, inputValues):
        #feed forward
        self.getOutput(inputValues)

        #change weights
        return self.calculateNewWeights()
        
    def calculateNewWeights(self):
        for i in range(0, self.LayerCount - 1):
            self.NewWeights[i] += sp.stats.zscore(self.Weights[i] * self.Neurons[i][np.newaxis].T) * self.Neurons[i+1].T 

        self.BatchSize += 1           

    def applyChanges(self, learningRate, biasLearningRate, regularization):
        #Changes the weights after a batch
        self.Weights = [(w + nw * (learningRate / self.BatchSize) - regularization * w) for w, nw in zip(self.Weights, self.NewWeights)]
        self.NewWeights = [np.zeros(x.shape) for x in self.NewWeights]
        
        self.BatchSize = 0            

class AverageLayer2D(BaseNetwork):
    def __init__(self):
        self.SlicesCount = 0
        self.SliceSize = 0 
    def getOutput(self, inputValues):  
        self.SlicesCount = inputValues.shape[0]      
        self.SliceSize = inputValues.shape[1]     
        return np.mean(inputValues, axis=1)
    def getDerivatives(self, outputDerivatives):
        return outputDerivatives / self.SliceSize
