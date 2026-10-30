import numpy as np

class Dense:
    """
    Dense Layer with Bias.
    """
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_inputs, n_outputs)
        # NEW: Initialize bias as zeros. Shape (1, n_outputs) for broadcasting.
        self.bias = np.zeros((1, n_outputs))

    def forward(self, X):
        self.last_input = X
        # NEW: Add bias to the dot product
        return np.dot(X, self.weights) + self.bias

    def backward(self, grad_output):
        # Blame for weights
        self.grad_weights = np.dot(self.last_input.T, grad_output)
        
        # NEW: Blame for bias. 
        # Since forward adds bias to every sample in the batch, 
        # the gradient for bias is the mean of grad_output across the batch (axis=0).
        self.grad_bias = np.mean(grad_output, axis=0, keepdims=True)
        
        # Pass complaint upstream
        return np.dot(grad_output, self.weights.T)

    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        # NEW: Update bias using its gradient
        self.bias -= learning_rate * self.grad_bias

class LeakyReLU:
    """
    Leaky ReLU: A bouncer that lets a tiny fraction of negative values pass.
    Prevents 'dead neurons' (Zombie Neurons) by ensuring gradients never hit absolute zero.
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, X):
        self.last_input = X
        # np.where(condition, value_if_true, value_if_false)
        return np.where(X > 0, X, self.alpha * X)

    def backward(self, grad_output):
        # Derivative is 1 if X > 0, else alpha (0.01)
        dx = np.where(self.last_input > 0, 1, self.alpha)
        return grad_output * dx
    
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

class Dropout:
    """
    Dropout Regularization: Randomly 'kills' neurons during training 
    to prevent the network from becoming over-reliant on specific neurons.
    """
    
    def __init__(self, keep_prob=0.8):
        """
        keep_prob: probability that a neuron is KEPT (not dropped).
        0.8 means 80% survive, 20% are killed.
        """
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, X, training=True):
        """
        Forward pass with dropout.
        
        During training: randomly drop neurons and scale up survivors.
        During evaluation/testing: no dropout applied.
        """
        if not training:
            return X
        
        # Create random mask: 1 = neuron survives, 0 = neuron killed
        self.mask = (np.random.rand(*X.shape) < self.keep_prob).astype(float)
        
        # Apply mask (killed neurons become 0)
        X_dropped = X * self.mask
        
        # Scale up survivors to maintain expected value (Inverted Dropout)
        return X_dropped / self.keep_prob

    def backward(self, grad_output):
        """
        Backward pass with dropout.
        
        Gradient only flows through neurons that survived.
        Killed neurons receive zero gradient.
        """
        # Safety check: if mask is None (eval mode), pass gradient through
        if self.mask is None:
            return grad_output
            
        # Apply mask to gradient (killed neurons get 0 gradient)
        return grad_output * self.mask / self.keep_prob