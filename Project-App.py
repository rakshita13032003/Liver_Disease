#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import numpy as np
import pickle
import pandas as pd

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

sex_val = 1 if sex == "Male" else 0

if st.button("Predict"):

    input_data = np.array([[age, sex_val, albumin, alk_phos, alt, ast,
                             bilirubin, cholinesterase, cholesterol,
                             creatinina, ggt, protein]])

    input_scaled = scaler.transform(input_data)

    proba = model.predict_proba(input_scaled)[0]
    classes = label_encoder.inverse_transform(np.arange(len(proba)))

    pred_index = np.argmax(proba)
    result = classes[pred_index]
    confidence = proba[pred_index]

    if result == "Cirrhosis" and confidence < 0.60:
        result = "Fibrosis"

    st.success(f"Predicted Liver Condition: **{result}**")
    st.info(f"Model Confidence: **{confidence*100:.2f}%**")

    st.subheader("Prediction Confidence")
    proba_df = pd.DataFrame({
        "Disease": classes,
        "Probability": proba
    })
    st.bar_chart(proba_df.set_index("Disease"))

    st.subheader("Clinical Interpretation")

    if alt > 55 or ast > 48:
        st.warning("Elevated ALT/AST indicates liver cell damage.")
    if bilirubin > 12:
        st.warning("High bilirubin suggests impaired liver function.")
    if albumin < 34:
        st.warning("Low albumin indicates reduced protein synthesis by the liver.")
    if ggt > 50:
        st.warning("High GGT indicates bile duct obstruction or alcohol-related damage.")
    if protein > 80:
        st.warning("Elevated protein may indicate kidney-liver interaction issues.")

