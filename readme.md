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
  - `ReLU` (and handling the "Dying ReLU" / Zombie Neuron problem).
  - `Softmax` for multi-class classification.
- **Loss Functions**: 
  - `MSE` (Mean Squared Error) for Regression.
  - `Cross-Entropy` for Classification.
- **Backpropagation Engine**: Manual implementation of the Chain Rule to compute gradients for every weight.
- **Optimizers**: Gradient Descent (SGD), with plans for Adam/AdamW.
- **Regularization**: Dropout, L2 Weight Decay, Early Stopping.

## 💡 Key Engineering Insights Discovered
Building this from scratch revealed the elegant simplicity beneath modern AI:
1. **The Chain Rule is just a messenger:** Backpropagation is simply passing the "blame" (gradient) backward through the pipes (weights), multiplied by the local derivative.
2. **ReLU is a bouncer:** It passes positive signals (derivative = 1) and kills negative ones (derivative = 0). If a neuron dies, its incoming weights freeze forever (Dying ReLU) because the gradient becomes exactly zero.
3. **Softmax + Cross-Entropy is a power couple:** Softmax forces raw scores (logits) into a 100% probability pie, and Cross-Entropy heavily penalizes the model if it's confidently wrong. The combined derivative simplifies beautifully to: `Gradient = Prediction - Target`.
4. **Synchronous Updates:** Gradients must be calculated using the *old* weights before any weight is updated. Updating weights mid-backprop breaks the chain rule.

## 🗺️ Learning Roadmap
This repo tracks my structured path from basic MLPs to modern architectures:
- [x] **Phase 1: Core Engine** (NumPy MLP, Backprop manual trace, Regresi vs Klasifikasi).
- [ ] **Phase 1.5: The MNIST Test** (Training on real-world image data, hyperparameter tuning, failure mode analysis).
- [ ] **Phase 2: PyTorch Transition** (Comparing my NumPy engine vs PyTorch's Autograd).

## 🚀 How to Run
*(Will be updated as the codebase grows)*
```bash
git clone https://github.com/riki/neural-network-from-satch.git
cd neural-network-from-scratch
python train.py
