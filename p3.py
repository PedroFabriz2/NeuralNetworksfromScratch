#same as p1.py but with numpy
import numpy as np

inputs = [3,2,1,3]
weights = [[1,2,3,4],
            [0,3.5,2,1],
            [2.3,1.2,3,4]]

biases = [2,3,4]

output = np.dot(weights, inputs)+biases
# weights comes first here because of what you want as output. In this case, we want to model 3 neurons receiveing 4 inputs
# so the output is align based on 3 sets of weights.

print(output)

