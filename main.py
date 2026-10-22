import numpy as np
from layers import Dense , ReLU

print("🚀 Neural Network From Scratch - Testing Dense Layer")
print("="*50)

# 1. INPUT DATA (X)
# Shape: 1 row, 2 columns (1 sample, 2 features: x1 and x2)
X = np.array([[2.0, 3.0]]) 

# 2. INSTANTIATE THE LAYER (2 inputs -> 2 hidden nodes)
layer1 = Dense(n_inputs=2, n_outputs=2)

# 3. INJECT MANUAL WEIGHTS
# Override the random weights to exactly match yesterday's manual calculation
layer1.weights = np.array([[ 2.0,  4.0],
                           [-1.0, -2.0]])

# 4. FORWARD PASS
H_raw = layer1.forward(X)
print(f"Raw Hidden Scores (Before Bouncer): {H_raw}") 

# 4. CALL THE BOUNCER (ReLU)
bouncer = ReLU()
H_active = bouncer.forward(H_raw)
print(f"Active Hidden Scores (After Bouncer): {H_active}")

# 5. LET'S TEST WITH NEGATIVE NUMBERS!
print("\n--- Testing Bouncer with Negative Numbers ---")
fake_negative_data = np.array([[-5.0, 2.0, -10.5, 8.0]])
print(f"Before Bouncer: {fake_negative_data}")
print(f"After Bouncer:  {bouncer.forward(fake_negative_data)}")

print("="*50)
print("✅ ReLU Bouncer successfully assembled!")