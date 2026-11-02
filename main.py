import numpy as np
import matplotlib.pyplot as plt
from layers import Dense, LeakyReLU, Dropout, Softmax, CrossEntropyLoss
from optimizers import AdamW
from sklearn.model_selection import train_test_split

print("🌀 GENERATING 3-CLASS SPIRALS DATASET...")

# ==========================================
# 1. GENERATE 3-CLASS SPIRALS
# ==========================================
def make_spirals(n_samples=300, n_classes=3, noise=0.2):
    """
    Generates interleaved spirals for multi-class classification.
    Each class forms one arm of the spiral.
    """
    # Create empty arrays to store coordinates (X) and labels (y)
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype='uint8')
    
    # Loop through each of the 3 classes (0, 1, 2)
    for class_num in range(n_classes):
        ix = range(n_samples * class_num, n_samples * (class_num + 1))
        
        # r (radius): distance from the center (0.0 to 1.0)
        r = np.linspace(0.0, 1, n_samples)
        
        # t (theta/angle): how far the arm wraps around the center
        # Each class starts at a different angle to avoid overlapping
        t = np.linspace(class_num * 4, (class_num + 1) * 4, n_samples) + np.random.randn(n_samples) * noise
        
        # Convert polar coordinates (r, t) to standard (x, y) coordinates
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_num
        
    return X, y

# Generate 300 points per class (Total = 900 points)
X, y = make_spirals(n_samples=300, n_classes=3, noise=0.2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Data Split Complete! Train: {len(X_train)} | Val: {len(X_val)}")

print(f"✅ Generated {len(X)} points across {len(np.unique(y))} classes!")

# ==========================================
# 2. VISUALIZATION (Dopamine Hit!)
# ==========================================
print("🎨 Plotting raw spirals... Close the window to continue.")
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=15)
plt.title("3-Class Spirals (The Multi-Class Battlefield)")
plt.xlabel("Feature 1 (X)")
plt.ylabel("Feature 2 (Y)")
plt.grid(True, alpha=0.3)
plt.show()

print("✅ Data prep complete! Ready for the next step.")

# ==========================================
# 3. MINI-BATCH ENGINE (Same as Two Moons!)
# ==========================================
def get_batches(X, y, batch_size=32):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0], batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield X[batch_indices], y[batch_indices]

# ==========================================
# 4. BUILD THE ARCHITECTURE (The Multi-Class Upgrade)
# ==========================================
# Input: 2 features (x, y)
# Hidden: 16 neurons (Space warping still needed!)
# Output: 3 classes (THIS IS THE ONLY CHANGE!)
hidden = Dense(n_inputs=2, n_outputs=32)
lrelu = LeakyReLU()
dropout = Dropout(keep_prob=0.8)
output = Dense(n_inputs=32, n_outputs=3)  # <--- UPGRADE TO 3 CLASSES!

softmax = Softmax()
ce_loss = CrossEntropyLoss()
optimizer = AdamW(learning_rate=0.01, weight_decay=0.01)

# ==========================================
# 5. THE TRAINING LOOPS (100% SAME AS TWO MOONS!)
# ==========================================
epochs = 100
batch_size = 32

print(f"\n🚀 STARTING MULTI-CLASS TRAINING ({epochs} Epochs)...")
print("-" * 50)

for epoch in range(epochs):
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
        
        # FORWARD PASS
        h_raw = hidden.forward(X_batch)
        h_active = lrelu.forward(h_raw)
        h_drop = dropout.forward(h_active, training=True)
        logits = output.forward(h_drop)
        probs = softmax.forward(logits)
        
        # LOSS & ACCURACY
        loss = ce_loss.forward(probs, y_batch)
        epoch_loss += loss * X_batch.shape[0]
        
        preds = np.argmax(probs, axis=1)
        correct_predictions += np.sum(preds == y_batch)
        total_samples += X_batch.shape[0]
        
        # BACKWARD PASS
        d_logits = ce_loss.backward()
        d_drop = output.backward(d_logits)
        d_active = dropout.backward(d_drop)
        d_raw = lrelu.backward(d_active)
        hidden.backward(d_raw)
        
        # UPDATE WEIGHTS
        optimizer.step(output)
        optimizer.step(hidden)
        
       # ==========================================
    # END OF EPOCH EVALUATION (The Overfitting Alarm)
    # ==========================================
    avg_loss = epoch_loss / total_samples
    
    # 1. CEK TRAIN ACCURACY (Dropout ON)
    h_train = hidden.forward(X_train)
    h_train_act = lrelu.forward(h_train)
    h_train_drop = dropout.forward(h_train_act, training=True)
    logits_train = output.forward(h_train_drop)
    probs_train = softmax.forward(logits_train)
    train_acc = np.mean(np.argmax(probs_train, axis=1) == y_train) * 100

    # 2. CEK VAL ACCURACY (Dropout OFF)
    h_val = hidden.forward(X_val)
    h_val_act = lrelu.forward(h_val)
    h_val_eval = dropout.forward(h_val_act, training=False) 
    logits_val = output.forward(h_val_eval)
    probs_val = softmax.forward(logits_val)
    val_acc = np.mean(np.argmax(probs_val, axis=1) == y_val) * 100

    # Print every 10 epochs
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

print("-" * 50)
print("✅ Training Complete! The spirals have been separated.")