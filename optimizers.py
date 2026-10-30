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
        learning_rate: seberapa besar langkah update (default PyTorch: 0.001)
        beta1: kekuatan memori momentum (0.9 = inget 90% arah sebelumnya)
        beta2: kekuatan memori variance (0.999 = inget 99.9% kekasaran jalan)
        epsilon: angka super kecil biar nggak kena division by zero
        weight_decay: kekuatan penalti (lambda). 0.01 = standar Transformer/LLM
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay  # ← INI yang bikin dia "AdamW"
        self.t = 0  # timestep counter
    
    def step(self, layer):
        """
        Update weights & biases untuk satu layer.
        
        PENTING: weight_decay TIDAK diaplikasikan ke bias!
        Ini best practice di hampir semua framework.
        """
        self.t += 1
        
        # ================================================================
        # STEP 1: UPDATE WEIGHTS (Pakai Adam)
        # ================================================================
        
        # Inisialisasi memory kalau belum ada (first time step)
        if not hasattr(layer, 'm_weights'):
            layer.m_weights = np.zeros_like(layer.weights)
            layer.v_weights = np.zeros_like(layer.weights)
            layer.m_bias = np.zeros_like(layer.bias)
            layer.v_bias = np.zeros_like(layer.bias)
        
        # --- MOMENTUM (m_t): "Bola Salju" ---
        # m_t = beta1 * m_t-1 + (1 - beta1) * gradient
        # Artinya: 90% arah sebelumnya + 10% arah gradient saat ini
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.grad_weights
        
        # --- RMSProp (v_t): "Sepatu Pintar" ---
        # v_t = beta2 * v_t-1 + (1 - beta2) * gradient^2
        # Kenapa dikuaratin? Karena kita cuma peduli BESAR variasi, bukan arahnya
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * (layer.grad_weights ** 2)
        
        # --- BIAS CORRECTION ---
        # Karena m & v mulai dari 0, di awal mereka "terlalu kecil"
        # Kita koreksi biar mereka akurat
        m_hat_w = layer.m_weights / (1 - self.beta1 ** self.t)
        v_hat_w = layer.v_weights / (1 - self.beta2 ** self.t)
        
        # --- ADAM UPDATE (bagian gradient descent biasa) ---
        # W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
        layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        
        # ================================================================
        # STEP 2: DECOUPLED WEIGHT DECAY (Ini yang bikin dia AdamW!)
        # ================================================================
        # W = W - lr * weight_decay * W
        # 
        # PERHATIKAN: Ini TERPISAH dari gradient di atas!
        # Nggak masuk ke m_t, nggak masuk ke v_t, nggak dibagi sqrt(v_hat)
        # Langsung aja kurangi weight-nya.
        # 
        # Efeknya: Weight yang GEDE tetap dipangkas dengan konsisten,
        # meskipun gradient-nya kecil.
        layer.weights -= self.lr * self.weight_decay * layer.weights
        
        # ================================================================
        # STEP 3: UPDATE BIAS (Cuma Adam, TANPA weight decay)
        # ================================================================
        # Bias nggak perlu di-penalty karena dia bukan "overfitting source"
        # Bias itu cuma "geser garis", bukan "slope" yang bikin overfit
        
        layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.grad_bias
        layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (layer.grad_bias ** 2)
        
        m_hat_b = layer.m_bias / (1 - self.beta1 ** self.t)
        v_hat_b = layer.v_bias / (1 - self.beta2 ** self.t)
        
        layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        # ← Liat? Nggak ada weight_decay di sini!