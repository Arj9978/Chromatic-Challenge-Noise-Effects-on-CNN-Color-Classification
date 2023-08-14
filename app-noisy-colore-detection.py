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

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(colors)
y = keras.utils.to_categorical(encoded_labels)

# User selects a color
selected_color = st.selectbox("Select a color", colors)

# Display the selected color
st.write(f"Selected Color: {selected_color}")

# Create a colored square image
def generate_color_image(color, size=(32, 32)):
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
# Display the colored square image using Streamlit
st.image(color_square, caption='Selected Color', channels='RGB')

def predict(): 
    color_square = np.array(color_square)
  
    # Get predicted probabilities for each class
    y_pred_probs = model.predict(color_square)
  
    # Get the predicted class labels by selecting the index of the maximum probability
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Convert encoded labels back to original labels
    y_pred_la = le.inverse_transform(y_pred)
    st.write(y_pred_la)



  
#     scaler = MinMaxScaler()
#     X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
#     X = np.array(X)
#     st.write(X)
#     prediction = model.predict(X)
#     if prediction[0] == 1: 
#         st.success('User Stay :thumbsup:')
#     else: 
#         st.error('User did not Stay :thumbsdown:')
#     st.write(prediction)

trigger = st.button('Predict', on_click=predict)
