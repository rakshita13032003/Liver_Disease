#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained objects
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Liver Disease Prediction", layout="centered")
st.title("🩺 Liver Disease Prediction System")

# User Inputs
age = st.number_input("Age", 1, 100)
sex = st.selectbox("Sex", ["Male", "Female"])
albumin = st.number_input("Albumin (g/L)")
alk_phos = st.number_input("Alkaline Phosphatase (U/L)")
alt = st.number_input("ALT (U/L)")
ast = st.number_input("AST (U/L)")
bilirubin = st.number_input("Bilirubin (mg/L)")
cholinesterase = st.number_input("Cholinesterase (U/L)")
cholesterol = st.number_input("Cholesterol (mmol/L)")
creatinina = st.number_input("Creatinina (µmol/L)")
ggt = st.number_input("GGT (IU/L)")
protein = st.number_input("Protein (mg)")

sex_val = 1 if sex == "Male" else 0

# Prediction
if st.button("Predict"):

    input_data = np.array([[age, sex_val, albumin, alk_phos, alt, ast,
                             bilirubin, cholinesterase, cholesterol,
                             creatinina, ggt, protein]])

    input_scaled = scaler.transform(input_data)

    # Probability prediction
    proba = model.predict_proba(input_scaled)[0]
    classes = label_encoder.inverse_transform(np.arange(len(proba)))

    # Initial ML prediction
    pred_index = np.argmax(proba)
    result = classes[pred_index]
    confidence = proba[pred_index]

    if result == "Cirrhosis":
        if albumin > 30 and bilirubin < 30 and ggt < 180:
            result = "Fibrosis"
    if result == "Fibrosis":
        if alt < 90 and ast < 90 and bilirubin < 25:
            result = "Hepatitis C"
    st.success(f"🩺 Predicted Liver Condition: **{result}**")
    st.info(f"Model Confidence: **{confidence*100:.2f}%**")
    st.subheader("Prediction Confidence Distribution")

    proba_df = pd.DataFrame({
        "Disease": classes,
        "Probability": proba
    })
    st.bar_chart(proba_df.set_index("Disease"))

    st.subheader("Clinical Interpretation")

    if alt > 55 or ast > 48:
        st.warning("Elevated ALT/AST indicates liver cell inflammation or damage.")

    if bilirubin > 12:
        st.warning("High bilirubin suggests impaired liver detoxification.")

    if albumin < 34:
        st.warning("Low albumin indicates reduced liver protein synthesis.")

    if ggt > 50:
        st.warning("High GGT suggests bile duct obstruction or chronic liver damage.")

    if protein > 80:
        st.warning("Elevated protein may indicate kidney–liver interaction issues.")

