#importing the Necessary Libraries

import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st

model = pkl.load(open('Insurance_ML.pkl', 'rb'))

st.header('Medical Insurance Premium Predictor')

gender = st.selectbox('Choose Gender', ['Female', 'Male'])
smoker = st.selectbox('Are you a smoker ?', ['Yes', 'No'])
region = st.selectbox('Choose Region', ['SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'])
age = st.slider('Enter Age', 5, 80)
bmi = st.slider('Enter BMI', 5, 100)
children = st.slider('Choose No of Children', 0, 5)

if st.button('Predict'):
    if gender == 'Female':
        gender = 1
    else:
        gender = 0

    if smoker == 'Yes':
        smoker = 1
    elif smoker == 'No':
        smoker = 0
    if region == 'SouthEast':
        region = 2
    if region == 'SouthWest':
        region = 3
    if region == 'NorthEast':
        region = 1
    if region == "NorthWest":
        region = 0

    input_data = (age, gender, bmi, children, smoker, region)
    input_data_array = np.asarray(input_data)
    input_data_array = input_data_array.reshape(1, -1)
    predicted_prem = model.predict(input_data_array)

    display_string = 'Insurance Premium will be ' + str(round(predicted_prem[0], 2)) + ' USD Dollars'

    st.markdown(display_string)
