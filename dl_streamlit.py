import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

import mlflow
import mlflow.keras

import streamlit as st

plt.style.use('seaborn')

@st.cache_resource
def load_models():
    dense_model = load_model('fashion_mnist_dense.keras')
    cnn_model = load_model('fashion_mnist_cnn.keras')
    return dense_model, cnn_model

@st.cache_resource
def load_data():
    (_, _), (X_test, y_test) = fashion_mnist.load_data()
    X_test = X_test.astype('float32') / 255.0
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                   
    return X_test, y_test, class_names

def preprocess_for_dense(image):
    return image.reshape(1, 28*28)

def preprocess_for_cnn(image):
    return image.reshape(1, 28, 28, 1)

def predict(image, model_type):
    dense_model, cnn_model = load_models()
    
    if model_type == 'Dense':
        processed_image = preprocess_for_dense(image)
        predictions = dense_model.predict(processed_image)
    else:  
        processed_image = preprocess_for_cnn(image)
        predictions = cnn_model.predict(processed_image)
        
    return predictions[0]

st.title('Projet Deep Learning - MNSIT Dataset')
X_test, y_test, class_names = load_data()
dense_model, cnn_model = load_models()

st.sidebar.title('Navigation')
app_mode = st.sidebar.radio('Choose the app mode',
    ['Model Exploration', 'Prediction Comparison'])

if app_mode == 'Model Exploration':
    st.header('Exploration des modèles')
    
    st.subheader('Architectures des modèles')
    
    st.write('Dense:')
    dense_summary = []
    dense_model.summary(print_fn=lambda x: dense_summary.append(x))
    st.text('\n'.join(dense_summary))
    
    st.write('CNN:')
    cnn_summary = []
    cnn_model.summary(print_fn=lambda x: cnn_summary.append(x))
    st.text('\n'.join(cnn_summary))
    
    st.subheader('Example Images from Fashion MNIST')
    
    num_rows = st.slider('Number of rows to display', 1, 5, 3)
    num_cols = st.slider('Number of columns to display', 1, 5, 3)
    
    indices = np.random.choice(len(X_test), num_rows * num_cols, replace=False)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i < len(axes):
            axes[i].imshow(X_test[idx], cmap='gray')
            axes[i].set_title(f'Class: {class_names[y_test[idx]]}')
            axes[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.header('Comparaison Prediction')
    
    selection_method = st.radio('Select an image by:', ['Random', 'Index'])
    
    if selection_method == 'Random':
        idx = np.random.randint(0, len(X_test))
    else:
        idx = st.slider('Select image index', 0, len(X_test)-1, 0)
    
    image = X_test[idx]
    true_class = y_test[idx]
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image, cmap='gray')
    ax.set_title(f'True Class: {class_names[true_class]}')
    ax.axis('off')
    st.pyplot(fig)
    
    dense_pred = predict(image, 'Dense')
    cnn_pred = predict(image, 'CNN')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Dense Prediction')
        dense_pred_class = np.argmax(dense_pred)
        dense_confidence = dense_pred[dense_pred_class] * 100
        
        st.write(f'Predicted Class: {class_names[dense_pred_class]}')
        st.write(f'Confidence: {dense_confidence:.2f}%')
        st.write(f'Correct: {"✓" if dense_pred_class == true_class else "✗"}')
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(class_names, dense_pred, color='skyblue')
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_title('Class Probabilities - Dense Model')
        ax.set_ylabel('Probability')
        
        bars[true_class].set_color('green')
        if dense_pred_class != true_class:
            bars[dense_pred_class].set_color('red')
            
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader('CNN Prediction')
        cnn_pred_class = np.argmax(cnn_pred)
        cnn_confidence = cnn_pred[cnn_pred_class] * 100
        
        st.write(f'Predicted Class: {class_names[cnn_pred_class]}')
        st.write(f'Confidence: {cnn_confidence:.2f}%')
        st.write(f'Correct: {"✓" if cnn_pred_class == true_class else "✗"}')
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(class_names, cnn_pred, color='skyblue')
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_title('Class Probabilities - CNN Model')
        ax.set_ylabel('Probability')
        
        bars[true_class].set_color('green')
        if cnn_pred_class != true_class:
            bars[cnn_pred_class].set_color('red')
            
        plt.tight_layout()
        st.pyplot(fig)