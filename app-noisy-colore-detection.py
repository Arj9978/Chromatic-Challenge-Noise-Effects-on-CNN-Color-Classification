# Import Libraries
import joblib
import sklearn

import numpy as np
import pandas as pd
import streamlit as st

from utils import PrepProcesor, columns 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

# Load Model
model = joblib.load('DecisionTreeClassifier.joblib')

# Define the unique addresses used for one-hot encoding during training
addresses = ['Shahran', 'Pardis', 'Shahrake Qods', 'Shahrake Gharb', 'North Program Organization', 'Andisheh', 'West Ferdows Boulevard', 'Narmak', 'Saadat Abad', 'Zafar', 'Islamshahr', 'Pirouzi', 'Shahrake Shahid Bagheri', 'Moniriyeh', 'Velenjak', 'Amirieh', 'Southern Janatabad', 'Salsabil', 'Zargandeh', 'Feiz Garden', 'Water Organization', 'ShahrAra', 'Gisha', 'Ray', 'Abbasabad', 'Ostad Moein', 'Farmanieh', 'Parand', 'Punak', 'Qasr-od-Dasht', 'Aqdasieh', 'Pakdasht', 'Railway', 'Central Janatabad', 'East Ferdows Boulevard', 'Pakdasht KhatunAbad', 'Sattarkhan', 'Baghestan', 'Shahryar', 'Northern Janatabad', 'Daryan No', 'Southern Program Organization', 'Rudhen', 'West Pars', 'Afsarieh', 'Marzdaran', 'Dorous', 'Sadeghieh', 'Chahardangeh', 'Baqershahr', 'Jeyhoon', 'Lavizan', 'Shams Abad', 'Fatemi', 'Keshavarz Boulevard', 'Kahrizak', 'Qarchak', 'Shahr-e-Ziba', 'Pasdaran', 'Northren Jamalzadeh', 'Azarbaijan', 'Bahar', 'Persian Gulf Martyrs Lake', 'Beryanak', 'Heshmatieh', 'Elm-o-Sanat', 'Golestan', 'Chardivari', 'Gheitarieh', 'Kamranieh', 'Gholhak', 'Heravi', 'Hashemi', 'Dehkade Olampic', 'Damavand', 'Republic', 'Zaferanieh', 'Qazvin Imamzadeh Hassan', 'Niavaran', 'Valiasr', 'Qalandari', 'Amir Bahador', 'Ekhtiarieh', 'Ekbatan', 'Absard', 'Haft Tir', 'Mahallati', 'Ozgol', 'Tajrish', 'Abazar', 'Koohsar', 'Hekmat', 'Parastar', 'Lavasan', 'Majidieh', 'Southern Chitgar', 'Karimkhan', 'Si Metri Ji', 'Karoon', 'Northern Chitgar', 'East Pars', 'Kook', 'Air force', 'Sohanak', 'Komeil', 'Azadshahr', 'Zibadasht', 'Amirabad', 'Dezashib', 'Elahieh', 'Mirdamad', 'Razi', 'Jordan', 'Mahmoudieh', 'Shahedshahr', 'Yaftabad', 'Mehran', 'Nasim Shahr', 'Tenant', 'Chardangeh', 'Fallah', 'Eskandari', 'Shahrakeh Naft', 'Ajudaniye', 'Tehransar', 'Nawab', 'Yousef Abad', 'Northern Suhrawardi', 'Villa', 'Hakimiyeh', 'Nezamabad', 'Garden of Saba', 'Tarasht', 'Azari', 'Shahrake Apadana', 'Araj', 'Vahidieh', 'Malard', 'Shahrake Azadi', 'Darband', 'Vanak', 'Tehran Now', 'Darabad', 'Eram', 'Atabak', 'Sabalan', 'SabaShahr', 'Shahrake Madaen', 'Waterfall', 'Ahang', 'Salehabad', 'Pishva', 'Enghelab', 'Islamshahr Elahieh', 'Ray - Montazeri', 'Firoozkooh Kuhsar', 'Ghoba', 'Mehrabad', 'Southern Suhrawardi', 'Abuzar', 'Dolatabad', 'Hor Square', 'Taslihat', 'Kazemabad', 'Robat Karim', 'Ray - Pilgosh', 'Ghiyamdasht', 'Telecommunication', 'Mirza Shirazi', 'Gandhi', 'Argentina', 'Seyed Khandan', 'Shahrake Quds', 'Safadasht', 'Khademabad Garden', 'Hassan Abad', 'Chidz', 'Khavaran', 'Boloorsazi', 'Mehrabad River River', 'Varamin - Beheshti', 'Shoosh', 'Thirteen November', 'Darakeh', 'Aliabad South', 'Alborz Complex', 'Firoozkooh', 'Vahidiyeh', 'Shadabad', 'Naziabad', 'Javadiyeh', 'Yakhchiabad']

st.title('Churn Modeling with Decision Trees')
# gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
# StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges, Churn, Contract_DSL, Contract_Fiber optic, Contract_No, PaymentMethod_Month-to-month, 
# PaymentMethod_One year, PaymentMethod_Two year, InternetService_Bank transfer (automatic), InternetService_Credit card (automatic), InternetService_Electronic check, 
# InternetService_Mailed check
st.code("0: No, 1: Yes")

Gender  = st.selectbox("Define The Gender (woman:0, man:1):", [0,1])
SeniorCitizen  = st.selectbox("Define SeniorCitizen Situation:", [0,1])
Partner  = st.selectbox("Do You have Partner:", [0,1])
Dependents  = st.selectbox("Do you have Dependents:", [0,1])
tenure  = st.slider("tenure",0,72)
PhoneService  = st.selectbox("PhoneService:", [0,1])
MultipleLines  = st.selectbox("MultipleLines:", [0,1])
OnlineSecurity  = st.selectbox("Choose OnlineSecurity:", [0,1])
OnlineBackup  = st.selectbox("Choose OnlineBackup:", [0,1])
DeviceProtection  = st.selectbox("Choose DeviceProtection:", [0,1])
TechSupport  = st.selectbox("Choose TechSupport:", [0,1])
StreamingTV  = st.selectbox("Choose StreamingTV:", [0,1])
StreamingMovies  = st.selectbox("Choose StreamingMovies:", [0,1])
PaperlessBilling  = st.selectbox("Choose PaperlessBilling:", [0,1])
MonthlyCharges  = st.slider("MonthlyCharges",18.25,118.75)
TotalCharges  = st.slider("TotalCharges",18.80,8684.80)
Contract_DSL  = st.selectbox("Choose Contract_DSL:", [0,1])
Contract_Fiber_optic  = st.selectbox("Choose Contract Fiber optic:", [0,1])
Contract_No  = st.selectbox("Choose Contract_No:", [0,1])
PaymentMethod_Month_to_month  = st.selectbox("Choose PaymentMethod-Month-to-month:", [0,1])
PaymentMethod_One_year  = st.selectbox("Choose PaymentMethod One year:", [0,1])
PaymentMethod_Two_year  = st.selectbox("Choose PaymentMethod Two year:", [0,1])
InternetService_Bank_transfer_automatic  = st.selectbox("Choose InternetService_Bank transfer (automatic):", [0,1])
InternetService_Credit_card_automatic  = st.selectbox("Choose InternetService_Credit card (automatic):", [0,1])
InternetService_Electronic_check  = st.selectbox("InternetService_Electronic check:", [0,1])
InternetService_Mailed_check  = st.selectbox("InternetService_Mailed check:", [0,1])


def predict(): 
    row = np.array([Gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                    StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges, Contract_DSL, Contract_Fiber_optic, Contract_No, 
                    PaymentMethod_Month_to_month, PaymentMethod_One_year, PaymentMethod_Two_year, InternetService_Bank_transfer_automatic, 
                    InternetService_Credit_card_automatic, InternetService_Electronic_check, InternetService_Mailed_check])

    # Create a DataFrame with the row data and columns matching the training data
    X = pd.DataFrame([row], columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 
                                     'Contract_DSL', 'Contract_Fiber optic', 'Contract_No', 'PaymentMethod_Month-to-month', 'PaymentMethod_One year',
                                     'PaymentMethod_Two year', 'InternetService_Bank transfer (automatic)', 'InternetService_Credit card (automatic)',
                                     'InternetService_Electronic check', 'InternetService_Mailed check'])

    cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
    scaler = MinMaxScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    X = np.array(X)
    st.write(X)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('User Stay :thumbsup:')
    else: 
        st.error('User did not Stay :thumbsdown:')
    st.write(prediction)

trigger = st.button('Predict', on_click=predict)
