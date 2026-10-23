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
    
class Softmax:
    """
    Softmax activation: converts raw logits into a probability
    distribution that sums to 1.0 (the 100% pie).
    """
    def forward(self, X):
        # Subtract the row max for numerical stability.
        # Why? e^1000 overflows float64 into 'inf'. Subtracting max
        # shifts scores down WITHOUT changing the final probabilities.
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_scores = np.exp(X_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
class CrossEntropyLoss:
    """
    Cross-Entropy loss for classification: punishes the model
    based on how confidently it was wrong about the true class.
    """
    def forward(self, probs, y_true):
        # Clip to avoid log(0) = -inf, which poisons training with NaN.
        probs_clipped = np.clip(probs, 1e-12, 1.0)
        # Grab the probability assigned to the TRUE class of each sample.
        correct_probs = probs_clipped[np.arange(len(y_true)), y_true]
        # Loss = mean negative log-likelihood.
        return np.mean(-np.log(correct_probs))