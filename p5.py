## NOTES: ##

## -----Activation Functions:-----
## " Why step functions to hidden layers?? "

## Step function; 
## ReLU function;
## Sigmond function;

## When I choose each type?
# https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5

# ------ ## Step function; 
#      Pros:

#      Cons:
# ------ ## ReLU function;
#      Pros: It is "almost linear" -> fast calcullations by your machine
            # The combination of various neurons can create a crazy function fit

#      Cons:
# ------ ## Sigmond function;
#      Pros:

#      Cons:

# ----------------------------------- #

# When we are using non-linear data, we need to add more hidden layers so our
#  combination of activation function works well



# ----------------------------------- #

# -------------- CODE ----------------#
import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


nnfs.init()

X,y = spiral_data(100,3)

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #the size here iis inverted so we dont need to transpose
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    

layer1 = Layer(2,5) # 3 because is the number of inputs in each batch and 10 is the number of neurons (like a hidden layer)

activation1 = ReLU() #creates an object from class ReLU (ReLU class doesnt have an init())

layer1.forward(X) #with the X as input we forward through the network

activation1.forward(layer1.output) # we take the output from the layer with 5 neurons and puts the results in an activation function

print(activation1.output)
        