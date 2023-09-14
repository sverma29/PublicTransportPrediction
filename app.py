# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:01:54 2023

@author: sverm
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import streamlit as st
import pandas as pd

# Load files
data = pd.read_csv("testFile.csv")
filename = "model0.sav"

X_test = data.drop(["Transport_Public Transport", "Work Exp"], axis=1)
y_test = data["Transport_Public Transport"]
print(X_test.head())

# Preprocess dataset



# load model with joblib
loaded_model = joblib.load(filename)


# Perform Streamlit operations
st.header("Public Transport Prediction")
age = st.number_input("Enter Age (in years)")
salary = st.number_input("Enter Salary (in thousands)")
distance = st.number_input("Enter Distance (in kms)")
genderStatus = st.radio("Select gender: ", ('Male', 'Female'))
licenseStatus = st.radio("Valid License?: ", ('Yes', 'No'))
educationStatus = st.multiselect("Select Education", ['Engineer', 'MBA'])

gender = 0
eng = 0
mba = 0
lic = 0
if (st.button('Predict')):
    if genderStatus == 'Male':
        gender = 1
    if 'Engineering' in educationStatus:
        eng = 1
    if 'MBA' in educationStatus:
        mba = 1
    if licenseStatus == "Yes":
        lic = 1
    
    input_data = pd.DataFrame([[age, salary, distance, gender, eng, mba, lic]],
                              columns=['Age', 'Salary', 'Distance', 'Gender_Male', 'Engineer', 'MBA', 'license'])
    prediction = loaded_model.predict(input_data)
    if prediction[0] == 1:
        st.write("The employee will use Public Transport")
    else:
        st.write("The employee will not use Public Transport")
    


## input_data = pd.DataFrame([age, salary, distance, gender_male, eng, mba, lic])
#input_data = pd.DataFrame([[26, 14.6, 11.6, 0, 1, 0, 0]], 
#                          columns=['Age', 'Salary', 'Distance', 'Gender_Male', 'Engineer', 'MBA', 'license'])
# y_predict = loaded_model.predict(input_data)
#print(y_predict)




# evaluate model 
#y_predict = loaded_model.predict(X_test)

# check results
#print(classification_report(y_test, y_predict))
