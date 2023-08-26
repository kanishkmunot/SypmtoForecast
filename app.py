import streamlit as st
import joblib
import numpy as np

# Load the pipeline
pipeline = joblib.load('model_pipeline.joblib')

def main():
    st.title('SymptoForecast: Disease Predictor')

    # Input fields for user to input data
    features = []  # List to hold feature values
    for i in range(394):
        feature = st.number_input(f'Feature {i + 1}:')
        features.append(feature)

    # Button to make predictions
    if st.button('Predict'):
        # Prepare input data for prediction
        input_data = np.array([features])  # Create a NumPy array

        # Make predictions using the pipeline
        prediction = pipeline.predict(input_data)

        # Display the prediction
        st.write(f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    main()
