## NOTES
# Batches: why use it?
# -> optimization and generalization. If your batch size is equal 1 sample, your model is trying to fit the data every sample
# However if you put a batch of 16 samples, your model will do less computation to the same amount of data at the end.
# Remember you dont want to cause overfitting while putting all the data at the same time.


#How to init your weights and biases
#weights are usually initialized from a range -1 to 1 and biases all zero sometimes
##-----------------------------------------------------------##


# Creating neurons as objects for OOP.
import numpy as np

X = [[0,4,2],[2,3,4],[1,2,3]]

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #the size here iis inverted so we dont need to transpose
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer(3,10) # 3 because is the number of inputs in each batch and 10 is the number of neurons (like a hidden layer)
layer2 = Layer(10,2) # 10 because the last layer has 10 neurons, so we actually have now 10 inputs and we want 2 output for each batch.

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)

