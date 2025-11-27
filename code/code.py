import numpy as np

inputs = [
    [1.2, 5.1, 2.1, 3],
    [3.2, 2.1, 8, 1],
    [2.2, 1.1, -1, 3],
    ]
weights = [[3.1, 5.4, 2.8, 1],
           [3.1, 5.4, 2.8, 2],
           [3.1, 5.4, 2.8, 4]]

bias = [2,3,4]


output = np.dot(inputs, np.array(weights).T) + bias
print(output)
