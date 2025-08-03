# Ayna ML Internship Assignment - Report

## 🔧 Hyperparameters
- Epochs: 20
- Batch Size: 16
- Learning Rate: 1e-3
- Loss Function: MSELoss
- Optimizer: Adam

## 🧠 Architecture
- UNet with 2 encoder-decoder blocks
- Color name embedded and concatenated as conditioning input

## 📉 Training Dynamics
- Train/Val loss decreased smoothly
- Output images gradually improved in clarity and color accuracy
- Failure cases: uncommon colors, complex polygons

## 🔍 Key Learnings
- Learned to condition vision models with embeddings
- Practiced augmenting input modalities
- Reinforced how to monitor training with wandb