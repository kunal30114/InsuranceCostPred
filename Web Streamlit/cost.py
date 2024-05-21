import streamlit as st
import numpy as np
import pickle

loaded_model = pickle.load(open('reg.pkl', 'rb'))

st.title("MEDICAL INSURANCE COST PREDICTION")

age = st.text_input('Enter your age:', '30')
gender = st.radio('Select your gender:', ('Male', 'Female'))
smoker = st.radio('Are you a smoker?', ('Yes', 'No'))
children = st.text_input('Enter number of children:', '0')
location = st.radio('Select your residence location:', ('Southeast', 'Southwest', 'Northeast', 'Northwest'))
bmi = st.text_input('Enter your BMI:', '25.0')

age = int(age)
gender = 0 if gender == 'Male' else 1
smoker = 0 if smoker == 'Yes' else 1
children = int(children)
location = ['Southeast', 'Southwest', 'Northeast', 'Northwest'].index(location)
bmi = float(bmi)


input_data = (age, gender, bmi, children, smoker, location)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = loaded_model.predict(input_data_reshaped)


if st.button("Evaluate"):
    cost = prediction[0]
    st.write('The estimated insurance cost is USD', cost)
