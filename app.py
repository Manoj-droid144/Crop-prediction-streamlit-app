import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import base64
import os

# Load dataset
data = pd.read_csv('Crop_Recommendation.csv')

# Preprocess dataset
X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
y = data['Crop']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #2980B9, #6DD5FA);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Crop Prediction App")
st.write("Enter the details of the land to predict the suitable crop.")

nitrogen = st.number_input("Nitrogen (N)")
phosphorus = st.number_input("Phosphorus (P)")
potassium = st.number_input("Potassium (K)")
temperature = st.slider("Temperature (°C)", int(data['Temperature'].min()), int(data['Temperature'].max()))
humidity = st.slider("Humidity (%)", int(data['Humidity'].min()), int(data['Humidity'].max()))
ph_value = st.slider("pH Value", float(data['pH_Value'].min()), float(data['pH_Value'].max()))
rainfall = st.slider("Rainfall (mm)", int(data['Rainfall'].min()), int(data['Rainfall'].max()))

if st.button("Predict Crop"):
    input_data = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]], 
                              columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])
    prediction = model.predict(input_data)
    st.write(f"Predicted Crop: {prediction[0]}")

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
