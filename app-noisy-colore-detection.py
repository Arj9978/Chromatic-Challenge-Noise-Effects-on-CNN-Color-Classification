# Import Libraries
import keras
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

# Load Model
model = keras.models.load_model("classification_model.h5")

# Streamlit app title and description
st.title("Chromatic Challenge: Noise Effects on CNN Color Classification")
st.write("Explore the impact of noise on color classification using a trained CNN model.")

# Define color labels
colors = ['blue', 'green', 'red', 'yellow', 'black', 'white']

# User selects a color
selected_color = st.selectbox("Select a color", colors)

# Display the selected color
st.write(f"Selected Color: {selected_color}")

# Create a colored square image
color_index = colors.index(selected_color)
color_square = np.ones((100, 100, 3)) * np.array(color_index) * 50 / 255.0
st.image(color_square, caption='Selected Color', channels='RGB')

# def predict(): 
#     row = np.array([Gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
#                     StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges, Contract_DSL, Contract_Fiber_optic, Contract_No, 
#                     PaymentMethod_Month_to_month, PaymentMethod_One_year, PaymentMethod_Two_year, InternetService_Bank_transfer_automatic, 
#                     InternetService_Credit_card_automatic, InternetService_Electronic_check, InternetService_Mailed_check])

#     # Create a DataFrame with the row data and columns matching the training data
#     X = pd.DataFrame([row], columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
#                                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 
#                                      'Contract_DSL', 'Contract_Fiber optic', 'Contract_No', 'PaymentMethod_Month-to-month', 'PaymentMethod_One year',
#                                      'PaymentMethod_Two year', 'InternetService_Bank transfer (automatic)', 'InternetService_Credit card (automatic)',
#                                      'InternetService_Electronic check', 'InternetService_Mailed check'])

#     cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
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

# trigger = st.button('Predict', on_click=predict)
