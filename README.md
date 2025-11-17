# Restricted Boltzmann Machine (RBM) on MNIST

This project implements a Restricted Boltzmann Machine (RBM) using TensorFlow and applies it for dimensionality reduction on the MNIST handwritten digits dataset. The RBM learns hidden features from the 784-dimensional images and reduces them into a 64/128-dimensional representation.

---

##  Objectives
- Load the MNIST dataset  
- Build an RBM with:
  - 784 visible units  
  - 64 or 128 hidden units  
  - Weight matrix + visible + hidden biases  
- Implement Contrastive Divergence (CD-1)  
- Train the RBM for 20+ epochs and track reconstruction error  
- Extract hidden features (reduced-dimensional representations)  
- Visualize the learned features using PCA or t-SNE  

---

## Project Structure
├── rbm.ipynb # Main notebook
├── README.md # Project documentation
└── requirements.txt # Dependencies

---

## Technologies Used
- Python  
- TensorFlow  
- NumPy  
- Matplotlib  
- scikit-learn  

---

##  Steps Performed

### **1. Load & preprocess MNIST**
Dataset loaded from `tensorflow.keras.datasets`.  
Images flattened from 28×28 → 784 and binarized for RBM training.

### **2. Build the RBM**
RBM includes:
- Weight matrix (784 × hidden_dim)  
- Visible bias  
- Hidden bias  
- Sigmoid activation  
- Bernoulli sampling  

### **3. Train using CD-1**
Contrastive Divergence (one Gibbs step) used to update:
- weights  
- visible biases  
- hidden biases  

Reconstruction error monitored each epoch.

### **4. Extract Hidden Features**
After training, hidden activations `p(h|v)` are used as the reduced features.

### **5. Visualize with PCA / t-SNE**
The reduced features (64/128 dims) are projected to 2D using:
- PCA  
- t-SNE  

Clusters show how well RBM learned patterns in MNIST digits.

---

##  Results
- Reconstruction error decreases across epochs  
- PCA and t-SNE show visible clusters for digits  
- RBM successfully learns meaningful hidden representations  

---


