import numpy as np

class Dense:
    """
    Dense Layer: A fully connected layer where every input node 
    is connected to every output node via a learnable weight.
    """
    def __init__(self, n_inputs, n_outputs):
        # Initialize weights randomly using a standard normal distribution.
        # Matrix shape: (number of inputs, number of outputs)
        self.weights = np.random.randn(n_inputs, n_outputs)

    def forward(self, X):
        # X = input data matrix (batch_size, n_inputs)
        # self.weights = weight matrix (n_inputs, n_outputs)
        # np.dot performs matrix multiplication (equivalent to manual Sigma sum)
        return np.dot(X, self.weights)
    

class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.
    Acts as a bouncer: passes positive values, blocks (zeros) negative values.
    """
    def forward(self, X):
        # np.maximum compares X and 0 element-wise, returning the larger value.
        # This elegantly replaces all negative numbers with 0 without slow for-loops.
        # Example: np.maximum([-5, 2], 0) -> [0, 2]
        return np.maximum(0, X)