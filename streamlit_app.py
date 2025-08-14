import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")
sex_encoder = joblib.load("sex_encoder.pkl")

st.title("Titanic Survival Prediction")

st.write("Enter passenger details to predict if they would survive.")

pclass = st.selectbox("Pclass (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)

if st.button("Predict Survival"):
    sex_encoded = sex_encoder.transform([sex])[0]
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex_encoded],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"Survived! (Probability: {probability:.2f})")
    else:
        st.error(f"Did not survive (Probability: {probability:.2f})")
