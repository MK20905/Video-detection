# 🎭 Deepfake Video Detection – MillionDeepfake Challenge 

This project tackles deepfake detection and localization using a multimodal model that combines video and audio inputs.

## 🧠 Goal
- Classify videos as real or fake
- Predict which video frames are manipulated

## ⚙️ Method
- **Video**: R3D-18 (spatio-temporal features from 32-frame clips)
- **Audio**: Multi-branch 1D-CNN (MFCC, chroma, spectral features, raw wave)
- **Fusion**: Concatenated embeddings → 8-head self-attention → dual outputs
- **Training**: BCE loss, AdamW, ReduceLROnPlateau, PyTorch custom pipeline

## 🔍 Features
- Visualization tools for per-frame prediction and spectrogram heatmaps
- Modular codebase for training, inference, and evaluation
