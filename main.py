import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from layers import Dense, LeakyReLU, Dropout, Softmax, CrossEntropyLoss
from optimizers import AdamW

print("🌕 GENERATING TWO MOONS DATASET...")

# ==========================================
# 1. GENERATE DATA
# ==========================================
# 1000 points, noise 0.1 to make it a bit messy (realistic), 
# random_state for reproducibility (same result every run)
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)


# ==========================================
# 3. MINI-BATCH ENGINE (The Kitchen Prep)
# ==========================================
def get_batches(X, y, batch_size=32):
    """
    Shuffles the data, then slices it into small batches.
    """
    # Create a list of indices from 0 to 999
    indices = np.arange(X.shape[0])
    
    # 🎲 SHUFFLE! Randomize indices so the model doesn't memorize the order
    np.random.shuffle(indices)
    
    # Slice into batches of size 32
    for start_idx in range(0, X.shape[0], batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield X[batch_indices], y[batch_indices]

# ==========================================
# 4. VISUALIZATION (Dopamine Hit!)
# ==========================================
print("🎨 Plotting raw data... Close the window to continue.")
plt.figure(figsize=(6, 4))
# Plot the X points. Color (c) is based on label y
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20)
plt.title("Raw Two Moons Data (The Battlefield)")
plt.xlabel("Feature 1 (X)")
plt.ylabel("Feature 2 (Y)")
plt.grid(True, alpha=0.3)
plt.show()

print("✅ Data prep complete! Ready for the Outer/Inner loops.")


# ==========================================
# 5. BUILD THE ARCHITECTURE (Space Warping Engine)
# ==========================================
# Input: 2 features (x, y coordinates)
# Hidden: 16 neurons (Need more brain power to bend space!)
# Output: 2 classes (Top Moon, Bottom Moon)
hidden = Dense(n_inputs=2, n_outputs=16)
lrelu = LeakyReLU()
dropout = Dropout(keep_prob=0.8)
output = Dense(n_inputs=16, n_outputs=2)

softmax = Softmax()
ce_loss = CrossEntropyLoss()

# Initialize AdamW (The Smart Optimizer)
optimizer = AdamW(learning_rate=0.01, weight_decay=0.01)

# ==========================================
# 6. THE TRAINING LOOPS (The Kitchen Cooking Process)
# ==========================================
epochs = 50
batch_size = 32

print(f"\n🚀 STARTING TRAINING ({epochs} Epochs, Batch Size {batch_size})...")
print("-" * 50)

# OUTER LOOP: Counting the Days (Epochs)
for epoch in range(epochs):
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # INNER LOOP: Cooking in Mini-Batches
    for X_batch, y_batch in get_batches(X, y, batch_size):
        
        # 1. FORWARD PASS (Prediction with Dropout ON)
        h_raw = hidden.forward(X_batch)
        h_active = lrelu.forward(h_raw)
        h_drop = dropout.forward(h_active, training=True) # 🎲 Random killing!
        logits = output.forward(h_drop)
        probs = softmax.forward(logits)
        
        # 2. CALCULATE LOSS
        loss = ce_loss.forward(probs, y_batch)
        epoch_loss += loss * X_batch.shape[0] # Accumulate loss weighted by batch size
        
        # Track Accuracy for this batch
        preds = np.argmax(probs, axis=1)
        correct_predictions += np.sum(preds == y_batch)
        total_samples += X_batch.shape[0]
        
        # 3. BACKWARD PASS (Compute Gradients)
        d_logits = ce_loss.backward()
        d_drop = output.backward(d_logits)
        d_active = dropout.backward(d_drop) # 🛡️ Mask the gradients!
        d_raw = lrelu.backward(d_active)
        hidden.backward(d_raw)
        
        # 4. UPDATE WEIGHTS (AdamW steps in)
        optimizer.step(output)
        optimizer.step(hidden)
        
    # ==========================================
    # END OF EPOCH METRICS (The Daily Report)
    # ==========================================
    avg_loss = epoch_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100
    
    # Print every 5 epochs to keep terminal clean, plus the final epoch
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

print("-" * 50)
print("✅ Training Complete! The space has been warped.")