# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
# torch.manual_seed(0)

# Here's a simple MLP
class MLPbrain():

    # Creates a random network
    @classmethod
    def randomBrain(self, size_inputs, size_layer1, size_out):
        square1 = np.sqrt(size_inputs)
        square2 = np.sqrt(size_layer1)
        # square3 = np.sqrt(size_layer2)
        weights = [np.random.uniform(low = -square1, high = square1, size=(size_inputs, size_layer1)), \
            np.random.uniform(low = -square2, high = square2, size=(size_layer1, size_out))]
            # np.random.uniform(low = -square3, high = square3, size=(size_layer2, size_out))]
        biases = [np.random.uniform(low = -square1, high = square1, size=(size_layer1)),\
            np.random.uniform(low = -square2, high = square2, size=(size_out))]
            # np.random.uniform(low = -square3, high = square3, size=(size_out))]
        return MLPbrain(weights + biases)

    # Creates the network from given weights
    def __init__(self, genes):

        self.weights = [genes[0], genes[1]]
        self.biases = [genes[2], genes[3]]
        
    # Does a forward pass over the network, processing the given inputs and returning the output
    def forward(self, inputs):
        layer1 = np.matmul(inputs, self.weights[0]) + self.biases[0]
        layer2 = np.matmul(layer1, self.weights[1]) + self.biases[1]
        # layer3 = np.matmul(layer2, self.weights[2]) + self.biases[2]
        layerOut = np.tanh(layer2)
        return layerOut
    
    # Applies a normally distributed random  
    def mutate(self, chance):
        for x in range(len(self.weights)):
            amplitude = np.sqrt(len(self.weights[x][0]))
            self.weights[x] += np.random.normal(0, amplitude, self.weights[x].shape) * np.random.choice(2, self.weights[x].shape, p=[1-chance, chance])
            self.weights[x] = np.clip(self.weights[x], -amplitude, amplitude) 
            # self.weights[x] = (self.weights[x] * amplitude) / self.getBiggestElement(self.weights[x]) #scale weights to intended range
            self.biases[x] += np.random.normal(0, amplitude, self.biases[x].shape) * np.random.choice(2, self.biases[x].shape, p=[1-chance, chance])
            self.biases[x] = np.clip(self.biases[x], -amplitude, amplitude) 
            # self.biases[x] = (self.biases[x] * amplitude) / self.getBiggestElement(self.biases[x])
        # print("yes sir")

    # Gets the biggest absolute value in the given matrix
    def getBiggestElement(self, matrix):
        return matrix.flat[abs(matrix).argmax()]
    
    def toString(self):
        return [self.weights, self.biases]
    
    @classmethod
    def crossover(self, brain1, brain2):
        shapes = [a.shape for a in brain1.weights + brain1.biases]
        # print(shapes)
                
        genes1 = np.concatenate([a.flatten() for a in brain1.weights + brain1.biases])
        genes2 = np.concatenate([a.flatten() for a in brain2.weights + brain2.biases])

        split = np.random.randint(0,len(genes1)-1)
        new_genes = np.asarray(genes1[0:split].tolist() + genes2[split:].tolist())
        new_genes = unflatten(new_genes, shapes)
        
        return MLPbrain(new_genes)
    
def unflatten(flattened,shapes):
        newarray = []
        index = 0
        for shape in shapes:
            size = np.product(shape)
            newarray.append(flattened[index : index + size].reshape(shape))
            index += size
        return newarray





# brain1 = createBrain()
# print(brain1.weights)
# print(brain1.forward([1,1,1,1,1,1,1,1,1,1]))

    # def forward(self, x):
    #     x = x.flatten(1)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.fc2(x)
    #     x = F.relu(x)
    #     x = self.fc3(x)
    #     return x
    
# robotBrain = SimpleMLP()


    # def initializeBrain():
    #     self.weights[0] = np.random.uniform(-)

    
    