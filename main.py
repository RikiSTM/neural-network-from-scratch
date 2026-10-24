import numpy as np
from layers import Dense, ReLU, Softmax, CrossEntropyLoss

print("🚀 NEURAL NETWORK TRAINING FROM SCRATCH")
print("="*50)

# 1. DATA & TARGET
X = np.array([[2.0, 3.0]])       # Input features: x1=2, x2=3
y_true = np.array([0])           # Target label: Cat (index 0)

# 2. BUILD ARCHITECTURE
hidden = Dense(n_inputs=2, n_outputs=2)
hidden.weights = np.array([[ 2.0,  4.0],
                           [-1.0, -2.0]]) # Initialize with manual weights for reproducibility

output = Dense(n_inputs=2, n_outputs=3)
output.weights = np.array([[ 1.0,  2.0,  3.0],
                           [-1.0,  0.0,  1.0]])

relu = ReLU()
softmax = Softmax()
ce_loss = CrossEntropyLoss()

# 3. HYPERPARAMETERS
learning_rate = 0.1
epochs = 100  # Number of training iterations

# 4. THE TRAINING LOOP (Where the magic happens)
for epoch in range(epochs):
    # --- FORWARD PASS (Prediction) ---
    h_raw = hidden.forward(X)
    h_active = relu.forward(h_raw)
    logits = output.forward(h_active)
    probs = softmax.forward(logits)
    loss = ce_loss.forward(probs, y_true)

    # --- BACKWARD PASS (Compute gradients) ---
    d_logits = ce_loss.backward()
    d_hidden = output.backward(d_logits)
    d_hidden_raw = relu.backward(d_hidden)
    hidden.backward(d_hidden_raw) # This populates hidden.grad_weights

    # --- UPDATE WEIGHTS (Apply Gradient Descent) ---
    output.update_params(learning_rate)
    hidden.update_params(learning_rate)

    # --- PRINT PROGRESS ---
    if epoch % 10 == 0 or epoch == epochs - 1:
        pred_class = np.argmax(probs)
        classes = ["Cat", "Dog", "Bird"]
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Prediction: {classes[pred_class]} (Probs: {np.round(probs, 4)})")

print("="*50)
print("✅ Training Complete! The network has learned.")