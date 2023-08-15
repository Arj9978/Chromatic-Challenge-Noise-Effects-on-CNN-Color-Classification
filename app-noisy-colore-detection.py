# Import Libraries
import keras
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from PIL import Image

from sklearn.preprocessing import LabelEncoder

# Load Model
model = keras.models.load_model("classification_model.h5")
  
# Streamlit app title and description
st.title("Chromatic Challenge: Noise Effects on CNN Color Classification")
st.write("Explore the impact of noise on color classification using a trained CNN model.")

# Define color labels
colors = ['blue', 'green', 'red', 'yellow', 'black', 'white']

# Define noise scale
sc = st.slider("scale",0,100)

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(colors)
y = keras.utils.to_categorical(encoded_labels)
st.write(y)

# User selects a color
selected_color = st.selectbox("Select a color", colors)

# Display the selected color
st.write(f"Selected Color: {selected_color}")

# Create a colored square image
# Create a colored square image
def generate_color_image(color, size=(32, 32)):  # Update image size to match model input size
    """Generate a monochrome image of the specified color."""
    color_map = {
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'red': (255, 0, 0),
        'yellow': (255, 255, 0),
        'black': (0, 0, 0),
        'white': (255, 255, 255)
    }
    
    # Ensure the color is valid
    if color not in color_map:
        raise ValueError(f"Unknown color {color}")
    
    # Create an image of the color
    img = Image.new('RGB', size, color_map[color])
    
    return img

color_square = generate_color_image(selected_color)
color_square_arr = np.array(color_square.resize((32, 32)))  # Resize image to match model input size
color_square_arr = color_square_arr / 255.0  # Normalize pixel values

# Display the colored square image using Streamlit
st.image(color_square, caption='Color without Noise', channels='RGB')

def predict(): 
    # Add Gaussian noise to the image
    noisy_color_square = color_square + np.random.normal(loc=0, scale=sc, size=color_square_arr.shape)
    noisy_color_square = np.clip(noisy_color_square, 0.0, 1.0)  # Clamp pixel values

    # Display the Noisy colored square image using Streamlit
    st.image(noisy_color_square, caption='Color with Noise', channels='RGB')

    # Resize the noisy color square image
    noisy_color_square_resized = noisy_color_square.resize((32, 32), Image.ANTIALIAS)

    # Convert the resized image to an array
    noisy_color_square_arr = np.array(noisy_color_square_resized)
    noisy_color_square_arr = noisy_color_square_arr / 255.0  # Normalize pixel values
  
    color_square_arr_expanded = np.expand_dims(noisy_color_square_arr, axis=0)  # Add batch dimension
    
    y_pred_probs = model.predict(color_square_arr_expanded)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_pred_la = le.inverse_transform(y_pred)
    st.write(y_pred_la)

trigger = st.button('Predict', on_click=predict)
