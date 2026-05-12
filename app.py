import streamlit as st
import pickle
import numpy as np

# Load saved model
model = pickle.load(open('logistic_model.pkl', 'rb'))

# Load scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title
st.title("Titanic Survival Prediction")

st.write("Enter passenger details below")

# Inputs
Pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

Sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

Age = st.slider(
    "Age",
    1,
    80,
    25
)

SibSp = st.number_input(
    "Siblings/Spouses",
    0,
    10
)

Parch = st.number_input(
    "Parents/Children",
    0,
    10
)

Fare = st.number_input(
    "Fare",
    0.0
)

Embarked = st.selectbox(
    "Embarked",
    ["C", "Q", "S"]
)

# Encoding
sex_encoded = 1 if Sex == "Male" else 0

embarked_map = {
    "C": 0,
    "Q": 1,
    "S": 2
}

embarked_encoded = embarked_map[Embarked]

# Predict Button
if st.button("Predict"):

    input_data = np.array([[
        Pclass,
        sex_encoded,
        Age,
        SibSp,
        Parch,
        Fare,
        embarked_encoded
    ]])

    # Scale data
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    # Output
    if prediction[0] == 1:
        st.success(
            f"Passenger Survived "
            f"(Probability: {probability:.2f})"
        )
    else:
        st.error(
            f"Passenger Did Not Survive "
            f"(Probability: {probability:.2f})"
        )