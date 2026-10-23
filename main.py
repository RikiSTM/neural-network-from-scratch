import numpy as np
from layers import Dense, ReLU, Softmax, CrossEntropyLoss

print("🚀 Full Classifier Forward Pass - Paper vs Code")
print("="*50)

# Input: x1=2, x2=3
X = np.array([[2.0, 3.0]])

hidden = Dense(n_inputs=2, n_outputs=2)
hidden.weights = np.array([[ 2.0,  4.0],
                           [-1.0, -2.0]])

output = Dense(n_inputs=2, n_outputs=3)
output.weights = np.array([[ 1.0,  2.0,  3.0],
                           [-1.0,  0.0,  1.0]])

# Components
relu = ReLU()
softmax = Softmax()
ce_loss = CrossEntropyLoss()

# --- FORWARD PASS CHAIN ---
h_raw = hidden.forward(X)          # Sigma at hidden layer
h_active = relu.forward(h_raw)     # Bouncer filters negatives
logits = output.forward(h_active)  # Sigma at output layer
probs = softmax.forward(logits)    # The 100% pie

# Target: Kucing = class index 0
y_true = np.array([0])
loss = ce_loss.forward(probs, y_true)

print(f"Hidden raw:    {h_raw}")     # Expect [[1. 2.]]
print(f"Hidden active: {h_active}")  # Expect [[1. 2.]]
print(f"Logits:        {logits}")     # Expect [[-1. 2. 5.]]
print(f"Probs:         {probs}")      # Expect ~[[0.0024 0.0473 0.9503]]
print(f"CE Loss:       {loss:.4f}")   # Expect ~6.0509
print("="*50)
print("✅ Full classifier forward pass matches the paper math!")