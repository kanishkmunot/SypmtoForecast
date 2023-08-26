
import streamlit as st
import joblib
import numpy as np

# Load the pipeline
pipeline = joblib.load('model_pipeline.joblib')

# Load the list of symptom names without the "Symptom_(no.)_" prefix
symptom_names = [' acidity', 'pain', 'discomfort', ' breathlessness', 'micturition', 'pain', ' chills', ' constipation', 'sneezing', ' cough', ' cramps', ' fatigue', ' headache', 'fever', ' indigestion', 'pain', 'swings', 'wasting', 'weakness', 'pain', 'movements', 'throat', 'pimples', ' shivering', 'rash', 'neck', 'pain', 'eyes', ' vomiting', 'limbs', 'gain', 'loss', 'skin', 'itching', 'pain', ' acidity', ' anxiety', ' blackheads', 'discomfort', ' blister', ' breathlessness', ' bruising', 'pain', ' chills', 'feets', ' cough', ' cramps', ' dehydration', ' dizziness', ' fatigue', 'of urine', ' headache', 'fever', ' indigestion', 'pain', 'pain', ' lethargy', 'appetite', 'swings', ' nausea', 'pain', 'eruptions', 'movements', 'region', 'throat', 'pimples', ' restlessness', ' shivering', 'peeling', 'rash', 'neck', 'pain', 'eyes', ' sweating', 'joints', 'tongue', ' vomiting', 'limbs', 'side', 'gain', 'loss', 'skin', 'pain', 'sensorium', ' anxiety', ' blackheads', ' blister', 'stool', 'vision', ' breathlessness', ' bruising', 'micturition', 'pain', ' chills', 'feets', 'urine', ' cough', 'urine', ' dehydration', ' diarrhoea', 'patches', ' dizziness', 'contacts', ' fatigue', 'of urine', ' headache', 'fever', 'pain', 'pain', 'pain', ' lethargy', 'appetite', 'balance', 'swings', 'stiffness', ' nausea', 'pain', 'eruptions', ' obesity', 'region', 'nose', ' restlessness', ' scurring', 'dusting', 'peeling', 'movements', 'pain', ' sweating', 'joints', 'stomach', 'tongue', ' vomiting', 'eyes', 'side', 'loss', 'skin', 'pain', 'sensorium', 'stool', 'vision', ' breathlessness', 'micturition', 'pain', 'urine', ' cough', 'urine', ' diarrhoea', 'patches', 'abdomen', ' dizziness', 'hunger', 'contacts', 'history', ' fatigue', ' headache', 'fever', 'pain', 'level', 'anus', 'concentration', ' lethargy', 'appetite', 'balance', 'swings', 'stiffness', ' nausea', ' obesity', 'walking', 'gases', 'nose', ' restlessness', ' scurring', 'dusting', 'nails', 'movements', ' urination', ' sweating', 'joints', 'stomach', 'legs', ' vomiting', 'eyes', 'loss', 'ooze', 'eyes', 'skin', 'pain', 'vision', ' breathlessness', 'pain', ' cough', 'urine', ' diarrhoea', 'abdomen', ' dizziness', 'hunger', 'history', ' fatigue', ' headache', 'fever', 'consumption', 'nails', 'itching', 'level', 'anus', 'concentration', ' lethargy', 'appetite', 'balance', 'sputum', ' nausea', 'walking', 'gases', 'nails', ' urination', 'neck', ' sweating', 'joints', 'vessels', 'legs', ' unsteadiness', 'ooze', 'eyes', 'skin', 'pain', 'vision', ' breathlessness', 'pain', ' constipation', 'urine', ' depression', ' diarrhoea', ' dizziness', 'history', 'rate', 'overload', ' headache', 'fever', 'consumption', 'nails', 'itching', 'appetite', ' malaise', 'sputum', ' nausea', ' obesity', 'walking', 'calf', 'eyes', 'neck', ' sweating', 'nodes', 'vessels', ' unsteadiness', 'eyes', 'skin', 'pain', 'vision', ' breathlessness', ' constipation', 'urine', ' depression', ' diarrhoea', 'thyroid', 'hunger', 'rate', 'overload', ' headache', ' irritability', 'appetite', ' malaise', 'fever', 'pain', ' nausea', ' obesity', ' phlegm', 'calf', 'eyes', ' sweating', 'nodes', 'urine', 'eyes', 'pain', 'nails', 'pain', ' diarrhoea', 'lips', 'thyroid', 'hunger', 'appetite', ' irritability', 'appetite', ' malaise', 'fever', 'pain', 'weakness', ' nausea', ' phlegm', ' sweating', 'nodes', 'disturbances', 'urine', 'eyes', 'pain', 'nails', 'pain', ' diarrhoea', 'lips', 'rate', 'appetite', ' irritability', 'appetite', ' malaise', 'fever', 'weakness', 'eyes', ' phlegm', ' polyuria', 'speech', 'nodes', 'extremeties', 'irritation', '(typhos)', 'disturbances', 'eyes', 'menstruation', 'failure', 'pain', 'pain', ' depression', 'rate', ' irritability', ' malaise', 'fever', 'pain', 'eyes', ' polyuria', 'transfusion', 'body', 'eyes', 'sputum', 'speech', 'extremeties', 'irritation', '(typhos)', 'eyes', 'menstruation', 'failure', 'pain', 'pain', ' coma', ' depression', ' irritability', ' malaise', 'pain', ' palpitations', 'transfusion', 'injections', 'body', 'eyes', 'sputum', 'pressure', 'nodes', 'eyes', 'menstruation', ' coma', ' irritability', ' malaise', 'pain', ' palpitations', 'injections', 'nose', 'pressure', 'bleeding', 'nodes', 'menstruation', ' congestion', ' malaise', 'pain', ' phlegm', 'body', 'nose', 'bleeding', 'pain', ' congestion', ' phlegm', 'body', 'sputum', 'pain', 'smell', 'sputum', 'smell', 'pain', 'pain']

def main():
    st.title('SymptoForecast: Disease Predictor')

    # Dropdown menu to select features
    selected_symptoms = st.multiselect('Select Features:', symptom_names)

    # Create an array to hold symptom values
    symptoms = []

    # Iterate through all symptom names
    for i, symptom_name in enumerate(symptom_names):
        unique_key = f'symptom_{i}'  # Generate a unique key for each input widget

        # If symptom_name is in the selected_symptoms list, create a number_input field
        if symptom_name in selected_symptoms:
            symptom_value = st.number_input(symptom_name.capitalize() + ':', key=unique_key, value=1, min_value=0, max_value=1, step=1)
        else:
            # If not selected, set the value to 0
            symptom_value = 0

        symptoms.append(symptom_value)

    # Button to make predictions
    if st.button('Predict'):
        # Prepare input data for prediction
        input_data = np.array([symptoms])  # Create a NumPy array

        # Make predictions using the pipeline
        prediction = pipeline.predict(input_data)

        # Display the prediction
        st.write(f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    main()


