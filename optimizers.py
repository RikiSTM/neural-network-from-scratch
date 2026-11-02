import numpy as np

class AdamW:
    """
    AdamW Optimizer: Adam + Decoupled Weight Decay.
    
    The KEY difference from Adam + L2:
    Weight decay is applied DIRECTLY to weights, 
    NOT passed through the gradient first.
    
    This ensures decay effect is CONSISTENT across all parameters,
    regardless of their gradient magnitude.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        """
        learning_rate: Step size for weight updates (PyTorch default: 0.001)
        beta1: Exponential decay rate for the 1st moment estimates (momentum)
        beta2: Exponential decay rate for the 2nd moment estimates (variance/RMSProp)
        epsilon: Small constant for numerical stability (prevents division by zero)
        weight_decay: Decoupled weight decay strength. 0.01 is the standard for Transformers/LLMs.
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay  # ← THIS is what makes it "AdamW"
        self.t = 0  # timestep counter
    
    def step(self, layer):
        """
        Update weights and biases for a single layer.
        
        IMPORTANT: weight_decay is NOT applied to biases!
        This is a best practice in almost all modern frameworks.
        """
        self.t += 1
        
        # ================================================================
        # STEP 1: UPDATE WEIGHTS (Using Adam)
        # ================================================================
        
        # Initialize momentum and variance buffers on the first step
        if not hasattr(layer, 'm_weights'):
            layer.m_weights = np.zeros_like(layer.weights)
            layer.v_weights = np.zeros_like(layer.weights)
            layer.m_bias = np.zeros_like(layer.bias)
            layer.v_bias = np.zeros_like(layer.bias)
        
        # --- MOMENTUM (m_t): The "Snowball" effect ---
        # m_t = beta1 * m_t-1 + (1 - beta1) * gradient
        # Meaning: 90% of previous direction + 10% of current gradient
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.grad_weights
        
        # --- VARIANCE / RMSProp (v_t): The "Smart Shoes" ---
        # v_t = beta2 * v_t-1 + (1 - beta2) * gradient^2
        # Why squared? We only care about the MAGNITUDE of variations, not the direction.
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * (layer.grad_weights ** 2)
        
        # --- BIAS CORRECTION ---
        # Since m & v start at 0, they are biased towards zero initially.
        # We correct them to be accurate.
        m_hat_w = layer.m_weights / (1 - self.beta1 ** self.t)
        v_hat_w = layer.v_weights / (1 - self.beta2 ** self.t)
        
        # --- ADAM UPDATE (Standard gradient descent step) ---
        # W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
        layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        
        # ================================================================
        # STEP 2: DECOUPLED WEIGHT DECAY (This is what makes it AdamW!)
        # ================================================================
        # W = W - lr * weight_decay * W
        # 
        # NOTICE: This is SEPARATED from the gradient above!
        # It doesn't go into m_t or v_t, and isn't divided by sqrt(v_hat).
        # We directly shrink the weights.
        # 
        # Effect: Large weights are consistently penalized, 
        # regardless of their gradient size.
        layer.weights -= self.lr * self.weight_decay * layer.weights
        
        # ================================================================
        # STEP 3: UPDATE BIAS (Adam only, NO weight decay)
        # ================================================================
        # Biases don't need penalties because they don't cause overfitting.
        # They just shift the line, they aren't the "slope" that memorizes noise.
        
        layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.grad_bias
        layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (layer.grad_bias ** 2)
        
        m_hat_b = layer.m_bias / (1 - self.beta1 ** self.t)
        v_hat_b = layer.v_bias / (1 - self.beta2 ** self.t)
        
        layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        # ← Notice? No weight_decay applied here!