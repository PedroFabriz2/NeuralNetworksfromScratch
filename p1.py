# common neuron : output = input*weight + bias

input = [1,2,3]
output = []
bias = 3
weights = [0.2,0.3,0.4]

# neuron:

output = input[0]*weights[0] + input[1]*weights[1] + input[2]*weights[2] + bias
#print(bias)

# This is basically what a neuron does in the neural network, we just have to put together a bunch of them.
# --------------------------------------------------------------------------------- #

# creating a loop for a layer calculation

inputs = [3,2,1,3]
weights = [[1,2,3,4],
            [0,3.5,2,1],
            [2.3,1.2,3,4]]

biases = [2,3,4]

layer_output = []

for w, b in  zip(weights, biases):
    output = 0
    for i, w_ in zip(inputs,w):
        output += i*w_
    output += b
    layer_output.append(output)

print(layer_output)

# --------------------------------------------------

