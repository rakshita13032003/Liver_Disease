#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("Liver Disease Prediction System")

age = st.number_input("Age", 1, 100)
sex = st.selectbox("Sex", ["Male", "Female"])
albumin = st.number_input("Albumin")
alk_phos = st.number_input("Alkaline Phosphatase")
alt = st.number_input("ALT")
ast = st.number_input("AST")
bilirubin = st.number_input("Bilirubin")
cholinesterase = st.number_input("Cholinesterase")
cholesterol = st.number_input("Cholesterol")
creatinina = st.number_input("Creatinina")
ggt = st.number_input("GGT")
protein = st.number_input("Protein")

# Manual mapping 
sex_val = 1 if sex == "Male" else 0

if st.button("Predict"):
    input_data = np.array([[age, sex_val, albumin, alk_phos, alt, ast,
                             bilirubin, cholinesterase, cholesterol,
                             creatinina, ggt, protein]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    result = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Liver Condition: {result}")


# In[ ]:




