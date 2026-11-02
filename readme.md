# 🧠 Neural Network From Scratch

> Unboxing the "black box" of Deep Learning using pure NumPy.

## 🎯 The "Why"
In an era where calling `model.fit()` in PyTorch or TensorFlow takes 3 lines of code, it's dangerously easy to treat Neural Networks as magic black boxes. This repository is my deliberate journey to strip away the abstractions and rebuild the core engine of Deep Learning from first principles—using nothing but Python and NumPy.

If you don't know how backpropagation flows through a computational graph, you can't debug why your loss is `NaN` or why your gradients are vanishing. This project is about owning the math.

## 🛠️ Core Implementations (The Engine)
Everything here is coded from scratch. No `autograd`, no `nn.Module`. Just raw matrix operations.

- **Tensor Operations**: Matrix multiplication (dot products) and broadcasting.
- **Forward Pass**: Data flowing through Multi-Layer Perceptrons (MLP).
- **Activation Functions**: 
  - `LeakyReLU` (Solving the "Dying ReLU" / Zombie Neuron problem by allowing a small gradient when negative).
  - `Softmax` for multi-class probability distributions.
- **Loss Functions**: 
  - `Cross-Entropy` paired with Softmax for Classification.
- **Backpropagation Engine**: Manual implementation of the Chain Rule to compute gradients for every weight and bias.
- **Optimizers**: 
  - Vanilla Gradient Descent (SGD).
  - **AdamW** (Adam with Decoupled Weight Decay for stable LLM-style training).
- **Regularization**: 
  - **Inverted Dropout** (Randomly killing neurons during training and scaling survivors to maintain expected energy).
  - Decoupled Weight Decay (via AdamW).
- **Training Loop**: Custom Mini-Batch Generator with data shuffling to prevent order-bias.

## 💡 Key Engineering Insights Discovered
Building this from scratch revealed the elegant simplicity beneath modern AI:

1. **The Chain Rule is just a messenger:** Backpropagation is simply passing the "blame" (gradient) backward through the pipes (weights), multiplied by the local derivative.
2. **Space Warping (The Manifold Hypothesis):** Hidden layers don't "think" like humans. They act as hands that stretch, fold, and warp the input space (like a rubber sheet) into higher dimensions. A non-linear problem (like Two Moons or Spirals) becomes linearly separable in this warped high-dimensional space, allowing the final linear layer to slice through it easily.
3. **ReLU is a bouncer:** It passes positive signals (derivative = 1) and kills negative ones. `LeakyReLU` fixes the "Dying ReLU" problem by letting a tiny gradient (e.g., 0.01) leak through, keeping the neuron alive.
4. **Softmax + Cross-Entropy is a power couple:** Softmax forces raw scores (logits) into a 100% probability pie, and Cross-Entropy heavily penalizes the model if it's confidently wrong. The combined derivative simplifies beautifully to: `Gradient = Prediction - Target`.
5. **The Magic of Mini-Batching:** Full-batch is too slow and gets stuck in local minima. Pure SGD (batch size 1) is too chaotic. Mini-batches (e.g., 32 samples) provide the perfect gradient signal-to-noise ratio for AdamW to navigate the loss landscape efficiently.
6. **Synchronous Updates:** Gradients must be calculated using the *old* weights before any weight is updated. Updating weights mid-backprop breaks the chain rule.

## 🗺️ Learning Roadmap
This repo tracks my structured path from basic MLPs to modern NLP architectures:

- [x] **Phase 1: Core Engine** (NumPy MLP, Backprop manual trace, AdamW, Dropout).
- [x] **Phase 1 Experiments: Space Warping** 
  - `Two Moons`: Binary classification & visualizing non-linear decision boundaries.
  - `Spirals`: Multi-class (3-class) classification & Softmax scaling.
- [ ] **Phase 1.5: The MNIST Test** (Handling high-dimensional inputs (784D) and real-world image data).
- [ ] **Phase 2: PyTorch Transition** (Fine-tuning BERT on Banking77 dataset for Intent Classification).
- [ ] **Phase 3: Transformer from Scratch** (Building Self-Attention and Positional Encodings without frameworks).

## 🚀 How to Run
This project uses `uv` for fast, reproducible Python environment management.

```bash
# Clone the repository
git clone https://github.com/riki/neural-network-from-scratch.git
cd neural-network-from-scratch

# Install dependencies (scikit-learn for datasets, matplotlib for visualization)
uv add scikit-learn matplotlib

# Run the current experiment (e.g., Two Moons or Spirals)
uv run main.py