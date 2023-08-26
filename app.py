import streamlit as st
import joblib
import pandas as pd

# Load the saved preprocessing pipeline and trained model
preprocessor = joblib.load('preprocessor.joblib')
trained_model = joblib.load('trained_model.joblib')

# List of possible symptoms (replace with your actual symptom list)
possible_symptoms = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17']

# Streamlit app code
st.title('Disease Predictor App')

# Multiselect widget for symptom selection
selected_symptoms = st.multiselect('Select Symptoms', possible_symptoms)

if st.button("Predict"):
    # Create a dictionary with selected symptoms
    input_data = {symptom: (1 if symptom in selected_symptoms else 0) for symptom in possible_symptoms}
    
    # Create an input DataFrame from the dictionary
    input_df = pd.DataFrame([input_data])

    # Preprocess input data using the loaded preprocessing pipeline
    preprocessed_data = preprocessor.transform(input_df)

    # Make prediction using the trained model
    prediction = trained_model.predict(preprocessed_data)

    st.write(f"Predicted Disease: {prediction[0]}")
