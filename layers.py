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
        # Save the input for the backward pass (the notebook pattern)
        self.last_input = X
        # np.dot performs matrix multiplication (equivalent to manual Sigma sum)
        return np.dot(X, self.weights)

    def backward(self, grad_output):
        """
        grad_output: complaint from downstream (gradient w.r.t this layer's output).
        Returns: complaint to pass upstream (gradient w.r.t this layer's input).
        """
        # Blame per weight = upstream activity × downstream complaint
        self.grad_weights = np.dot(self.last_input.T, grad_output)
        # Redistribute downstream complaints back upstream through transposed pipes
        return np.dot(grad_output, self.weights.T)
    
    def update_params(self, learning_rate):
        """
        Apply Gradient Descent: adjust weights in the opposite direction of the gradient.
        W_new = W_old - (lr * grad)
        """
        self.weights -= learning_rate * self.grad_weights
    

class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.
    Acts as a bouncer: passes positive values, blocks (zeros) negative values.
    """
    def forward(self, X):
        # Save the input mask for backward: remember who got kicked out
        self.last_input = X
        return np.maximum(0, X)

    def backward(self, grad_output):
        # Gate the gradient: flows only where the forward input was positive.
        # (last_input > 0) builds a mask of 1s (passed) and 0s (blocked).
        return grad_output * (self.last_input > 0)
    
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
        
        # Save state for the backward pass
        self.probs = probs
        self.y_true = y_true
        # Clip to avoid log(0) = -inf, which poisons training with NaN.
        probs_clipped = np.clip(probs, 1e-12, 1.0)
        # Grab the probability assigned to the TRUE class of each sample.
        correct_probs = probs_clipped[np.arange(len(y_true)), y_true]
        # Loss = mean negative log-likelihood.
        return np.mean(-np.log(correct_probs))
    
    def backward(self):
        """
        Computes the gradient of the loss w.r.t the logits (pre-softmax).
        The magical combined derivative of Softmax + CE is simply: probs - one_hot.
        """
        # Create a copy of probs so we don't mutate the original array
        grad = np.copy(self.probs)
        
        # Subtract 1 from the probability of the TRUE class (One-Hot subtraction)
        grad[np.arange(len(self.y_true)), self.y_true] -= 1
        
        # Divide by batch size because the forward pass used np.mean()
        return grad / len(self.y_true)
    