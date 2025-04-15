# fashion-mnist-streamlit-app
Deep learning classification of Fashion MNIST dataset with Streamlit interface and MLflow tracking.
This project applies deep learning techniques to the Fashion MNIST dataset, allowing interactive comparison of a Dense and a CNN model through a Streamlit interface.

## Features
- TensorFlow/Keras models: one Dense and one CNN
- Visual model architecture display
- Interactive comparison of prediction performance
- Integrated MLflow experiment tracking
- Streamlit-based user interface

## File Structure
- `projet_DL.ipynb`: Data exploration and model training
- `dl_streamlit.py`: Streamlit app for inference and visualization
- `fashion_mnist_dense.keras`: Dense model saved in Keras format
- `fashion_mnist_cnn.keras`: CNN model saved in Keras format

## Installation
```bash
pip install -r requirements.txt
