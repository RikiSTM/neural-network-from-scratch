import numpy as np

class Adam:
    """
    Adam Optimizer: Adaptive Moment Estimation.
    An upgrade to standard Gradient Descent that tracks momentum 
    and adapts the learning rate for every single parameter.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        
        # beta1 controls momentum (how much we remember past gradients)
        self.beta1 = beta1
        
        # beta2 controls adaptive learning rate (how much we track gradient variance)
        self.beta2 = beta2
        
        # epsilon is a tiny number to prevent division by zero
        self.epsilon = epsilon
        
        # t is the time step (iteration count), used to correct initial bias
        self.t = 0

    def step(self, layer):
        """
        Applies the Adam update rule to a Dense layer's weights and biases.
        This IS Gradient Descent, but with smart memory.
        """
        self.t += 1
        
        # 1. Initialize memory variables (m and v) if they don't exist in the layer yet
        if not hasattr(layer, 'm_weights'):
            layer.m_weights = np.zeros_like(layer.weights)
            layer.v_weights = np.zeros_like(layer.weights)
            layer.m_bias = np.zeros_like(layer.bias)
            layer.v_bias = np.zeros_like(layer.bias)

        # --- UPDATE FOR WEIGHTS ---
        # Momentum (m): Exponential moving average of gradients (The Snowball)
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.grad_weights
        
        # Adaptive LR (v): Exponential moving average of SQUARED gradients (Smart Shoes)
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * (layer.grad_weights ** 2)

        # Bias correction for m and v (since they started as zeros)
        m_hat_w = layer.m_weights / (1 - self.beta1 ** self.t)
        v_hat_w = layer.v_weights / (1 - self.beta2 ** self.t)

        # THE CORE GRADIENT DESCENT UPDATE:
        # W = W - (lr * momentum) / (sqrt(variance) + epsilon)
        layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)


        # --- UPDATE FOR BIAS ---
        layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.grad_bias
        layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (layer.grad_bias ** 2)

        m_hat_b = layer.m_bias / (1 - self.beta1 ** self.t)
        v_hat_b = layer.v_bias / (1 - self.beta2 ** self.t)

        layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)