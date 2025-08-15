import streamlit as st
import pickle
import os
import pandas as pd

st.title("🚢 SurviveTitan - Titanic Survival Prediction")

# Check if model exists
model_path = "model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please run `python train.py` first to train and save the model.")
    st.stop()

# Load trained model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Input form
st.header("Enter Passenger Details")
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, value=30.0)

# Convert inputs to dataframe
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [0 if sex == "male" else 1],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("🎉 This passenger would have SURVIVED!")
    else:
        st.error("💀 This passenger would NOT have survived.")
