# Simple-RNN-model

🎬 Movie Review Sentiment Analysis using SimpleRNN

This project is a Sentiment Analysis Web App built using a Simple Recurrent Neural Network (RNN).
It classifies movie reviews as Positive 😊 or Negative 😞.

🚀 Features
Predict sentiment of user-input movie reviews
Built using Deep Learning (RNN)
Interactive UI using Streamlit
Real-time prediction with probability score
Text preprocessing with tokenization & padding

🧠 Model Details
Model Type: SimpleRNN
Output: Binary classification (Positive / Negative)
Activation: Sigmoid
Loss Function: Binary Crossentropy
Optimizer: Adam

🛠️ Tech Stack
Python 🐍
TensorFlow / Keras
Streamlit
NumPy
Pickle (for tokenizer/model)

## 🚀 Live Demo

This project is deployed and accessible online:
https://rnn-basic.streamlit.app/

## 🖥️ Screenshots

<p align="center">
  <img src="https://raw.githubusercontent.com/Riitwiik/Simple-RNN-model/refs/heads/main/Negative.png" width="45%"/>
  <img src="https://raw.githubusercontent.com/Riitwiik/Simple-RNN-model/refs/heads/main/Positive.png" width="45%"/>
</p>

## 🛡️ Regularization Techniques

- **Dropout**: Randomly drops neurons during training to prevent over-reliance on specific features.
- **L2 Regularization**: Penalizes large weights to reduce model complexity and improve generalization.
