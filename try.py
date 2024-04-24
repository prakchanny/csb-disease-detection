import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pandas as pd
import random

# Function to load the models
def load_models():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
    return diabetes_model, heart_disease_model

# Function for Diabetes Prediction
def diabetes_prediction(user_input, diabetes_model):
    diabetes_prediction = diabetes_model.predict([user_input])
    return diabetes_prediction[0]  # Return 1 if diabetic, 0 if not diabetic

# Function for Heart Disease Prediction
def heart_disease_prediction(user_input, heart_disease_model):
    heart_prediction = heart_disease_model.predict([user_input])
    return heart_prediction[0]  # Return 1 if having heart disease, 0 if not

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Medical Diagnosis Assistant", layout="wide", page_icon="🩺")

    # Load machine learning models
    diabetes_model, heart_disease_model = load_models()

    # Sidebar for navigation
    with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',
                               ['Diabetes Prediction', 'Heart Disease Prediction'],
                               menu_icon='cross', icons=['activity', 'heart', 'person'], default_index=0)

    # Define threshold values for alarming features
    threshold_values = {
        'Pregnancies': 5,  # Example threshold value, adjust as needed
        'Glucose Level': 140,
        'Blood Pressure value': 90,
        'Skin Thickness value': 35,
        'Insulin Level': 100,
        'BMI value': 30,
        'Diabetes Pedigree Function value': 1.0,
        'Age of the Person': 60
    }

    # Diabetes Prediction Page
    if selected == 'Diabetes Prediction':
        st.title('Diabetes Prediction using Machine Learning')
        # Input fields
        pregnancies = st.slider('Number of Pregnancies', min_value=0, max_value=20, value=0, format="%d")
        glucose = st.slider('Glucose Level', min_value=0, max_value=300, value=0, format="%d mg/dl")
        blood_pressure = st.slider('Blood Pressure value', min_value=0, max_value=200, value=0, format="%d mmHg")
        skin_thickness = st.slider('Skin Thickness value', min_value=0, max_value=100, value=0, format="%d mm")
        insulin = st.slider('Insulin Level', min_value=0, max_value=300, value=0, format="%d mU/ml")
        bmi = st.slider('BMI value', min_value=0.0, max_value=50.0, value=0.0, format="%.1f")
        diabetes_pedigree_function = st.slider('Diabetes Pedigree Function value', min_value=0.0, max_value=2.5, value=0.0, format="%.1f")
        age = st.slider('Age of the Person', min_value=0, max_value=150, value=0, format="%d years")

        # Prediction button
        if st.button('Diabetes Test Result'):
            user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
            result = diabetes_prediction(user_input, diabetes_model)
            if result == 1:
                st.success("The person is diabetic")
            else:
                st.success("The person is not diabetic")

            # Calculate the percentage of users diagnosed with heart disease
            percentage_heart_disease = calculate_percentage_heart_disease(heart_disease_model)
            st.write(f"Percentage of users diagnosed with heart disease: {percentage_heart_disease:.2f}%")

            # Visualize the input data and alarming values
            visualize_input_and_thresholds(user_input, threshold_values)

    # Heart Disease Prediction Page
    if selected == 'Heart Disease Prediction':
        st.title('Heart Disease Prediction using Machine Learning')
        # Input fields
        age = st.slider('Age', min_value=0, max_value=150, value=0, format="%d years")
        sex = st.radio('Sex', ['Male', 'Female'])
        # Map sex to numeric value
        sex = 1 if sex == 'Male' else 0
        cp = st.radio('Chest Pain types', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
        # Map cp to numeric value
        cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
        cp = cp_mapping[cp]
        trestbps = st.slider('Resting Blood Pressure', min_value=0, max_value=300, value=0, format="%d mmHg")
        chol = st.slider('Serum Cholestoral in mg/dl', min_value=0, max_value=600, value=0, format="%d mg/dl")
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
        # Map fbs to numeric value
        fbs = 1 if fbs == 'Yes' else 0
        restecg = st.radio('Resting Electrocardiographic results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        # Map restecg to numeric value
        restecg_mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
        restecg = restecg_mapping[restecg]
        thalach = st.slider('Maximum Heart Rate achieved', min_value=0, max_value=300, value=0, format="%d bpm")
        exang = st.radio('Exercise Induced Angina', ['Yes', 'No'])
        # Map exang to numeric value
        exang = 1 if exang == 'Yes' else 0
        oldpeak = st.slider('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=0.0, format="%.1f mm")
        slope = st.radio('Slope of the peak exercise ST segment', ['Upsloping', 'Flat', 'Downsloping'])
        # Map slope to numeric value
        slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
        slope = slope_mapping[slope]
        ca = st.select_slider('Major vessels colored by flourosopy', options=['0', '1', '2', '3'], value='0', format_func=lambda x: f'{x} vessels')
        # Map ca to numeric value
        ca = int(ca)
        thal = st.radio('Thal', ['Normal', 'Fixed Defect', 'Reversable Defect'])
        # Map thal to numeric value
        thal_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversable Defect': 2}
        thal = thal_mapping[thal]

        # Threshold values for alarming features
        threshold_values = {
            'Age': 50,  # Example threshold value, adjust as needed
            'Resting BP': 140,
            'Cholesterol': 200,
            'Max HR': 160
        }

        # Prediction button
        if st.button('Heart Disease Test Result'):
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            result = heart_disease_prediction(user_input, heart_disease_model)
            if result == 1:
                st.success("The person is having heart disease")
            else:
                st.success("The person does not have any heart disease")

            # Calculate the percentage of users diagnosed with heart disease
            percentage_heart_disease = calculate_percentage_heart_disease(heart_disease_model)
            st.write(f"Percentage of users diagnosed with heart disease: {percentage_heart_disease:.2f}%")

            # Visualize the input data and alarming values
            visualize_input_and_thresholds(user_input, threshold_values)

def calculate_percentage_heart_disease(heart_disease_model, num_samples=1000):
    diagnosed_count = 0
    for _ in range(num_samples):
        # Generate random input data for testing
        user_input = [random.randint(0, 150) for _ in range(13)]
        result = heart_disease_model.predict([user_input])[0]
        diagnosed_count += result
    percentage = (diagnosed_count / num_samples) * 100
    return percentage

def visualize_input_and_thresholds(user_input, threshold_values):
    # Visualize the input data
    df_input = pd.DataFrame({'Feature': ['Pregnancies', 'Glucose Level', 'Blood Pressure value', 'Skin Thickness value',
                                         'Insulin Level', 'BMI value', 'Diabetes Pedigree Function value', 'Age of the Person'],
                             'Value': user_input})
    st.write("### Input Features")
    st.table(df_input)

    # Visualize the alarming values based on thresholds
    df_thresholds = pd.DataFrame({'Feature': list(threshold_values.keys()),
                                  'Threshold': list(threshold_values.values())})
    st.write("### Alarming Thresholds")
    st.table(df_thresholds)

    # Plot the input data and alarming values
    fig, ax = plt.subplots(figsize=(10, 6))
    features = df_input['Feature']
    values = df_input['Value']
    thresholds = df_thresholds['Threshold']
    colors = ['red' if val >= 20 else 'green' if val <= threshold else 'blue' for val, threshold in zip(values, thresholds)]
    ax.barh(features, values, color=colors)
    ax.set_xlabel('Value')
    ax.set_ylabel('Feature')
    ax.set_title('Input Features and Alarming Thresholds')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
