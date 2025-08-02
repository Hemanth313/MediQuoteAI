import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load models
with open("polynomial_regression.pkl", "rb") as f:
    polynomial_regression = pickle.load(f)

with open("family_floater.pkl", "rb") as f:
    family_floater = pickle.load(f)

def predict_medical_insurance_cost(model, age, sex, bmi, children, smoker, region, parents=0):
    scaler = StandardScaler()

    if model == "Family Floater":
        features = np.array([[age, sex, bmi, children, smoker, region, parents]])
        new_features_df = pd.DataFrame(features, columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'parents'])
        new_features_df[['age', 'bmi']] = scaler.fit_transform(new_features_df[['age', 'bmi']])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        new_poly_features = poly.fit_transform(new_features_df[['age', 'bmi']])
        poly_feature_names = poly.get_feature_names_out(['age', 'bmi'])
        new_poly_df = pd.DataFrame(new_poly_features, columns=poly_feature_names)
        new_X_poly = pd.concat([new_features_df.drop(columns=['age', 'bmi']), new_poly_df], axis=1)
        prediction = family_floater.predict(new_X_poly)

    else:
        features = np.array([[age, sex, bmi, children, smoker, region]])
        new_features_df = pd.DataFrame(features, columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        new_features_df[['age', 'bmi']] = scaler.fit_transform(new_features_df[['age', 'bmi']])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        new_poly_features = poly.fit_transform(new_features_df[['age', 'bmi']])
        poly_feature_names = poly.get_feature_names_out(['age', 'bmi'])
        new_poly_df = pd.DataFrame(new_poly_features, columns=poly_feature_names)
        new_X_poly = pd.concat([new_features_df.drop(columns=['age', 'bmi']), new_poly_df], axis=1)
        prediction = polynomial_regression.predict(new_X_poly)
    
    return prediction

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #0f0f1c;
            color: white;
        }
        .main {
            background-color: #0f0f1c;
            padding: 20px;
        }
        .header {
            background-color: #0dcaf0;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h3 {
            color: #0f0f1c;
            text-align: center;
        }
        .form-container {
            background-color: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        .form-container input, .form-container select {
            color: black;
        }
        .result-box {
            background-color: #17c964;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            margin-top: 20px;
        }
        .about-box {
            background-color: #3f51b5;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
        .footer {
            text-align: center;
            font-size: 16px;
            margin-top: 30px;
            color: #bbb;
        }
        .welcome-section {
            background:#1a1a2e;
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 30px;
            border: solid 2px #0dcaf0;
        }
        .welcome-section h1 {
            font-size: 2.5rem;
            text-align: center;
            color: #0dcaf0;
        }
        .welcome-section p {
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App
def main():
    # Welcome section
    st.markdown("""
    <div class="welcome-section">
        <h1>Welcome to MediQuote AI</h1>
        <p>
            Predicting medical insurance premiums helps ensure fairness and accuracy in pricing.
            Insurers can assess financial risks effectively, and individuals get transparent costs based on real data factors like age, lifestyle, and location.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header"><h3>Medical Insurance Premium Cost Prediction</h3></div>', unsafe_allow_html=True)

    model_type = st.selectbox("Choose Model Type", ["Polynomial Model", "Family Floater"])
    height_unit = st.selectbox("Height Unit", ["centimeter", "feet and inches"])

    with st.form(key="insurance_form"):
        age = st.number_input("Age (in years)", min_value=1, max_value=100, value=25)
        sex = st.radio("Sex", ["Male", "Female"])
        weight = st.number_input("Weight (in kg)", value=70.0)

        if height_unit == "centimeter":
            height = st.number_input("Height (in cm)", value=170.0)
        else:
            col1, col2 = st.columns(2)
            with col1:
                feet = st.number_input("Height (Feet)", min_value=0, max_value=8, value=5)
            with col2:
                inches = st.number_input("Height (Inches)", min_value=0.0, max_value=11.9, value=8.0)
            height = feet * 30.48 + inches * 2.54

        bmi = weight / (height / 100) ** 2
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.radio("Are you a smoker?", ["Yes", "No"])
        region = st.radio("Select your region", ["North-East", "North-West", "South-East", "South-West"])

        parents = 0
        if model_type == "Family Floater":
            parents = st.number_input("Number of Parents", min_value=0, max_value=2, value=0)

        submit = st.form_submit_button("Predict Insurance Premium")

    if submit:
        sex = 1 if sex == "Male" else 0
        smoker = 1 if smoker == "Yes" else 0
        region_map = {"North-East": 0, "North-West": 1, "South-East": 2, "South-West": 3}
        region = region_map[region]

        if not (1 <= age <= 100) or not (0 <= bmi <= 100):
            st.error("Invalid age or BMI input.")
        else:
            try:
                prediction = predict_medical_insurance_cost(model_type, age, sex, bmi, children, smoker, region, parents)[0]
                if prediction < 1000:
                    prediction = 1000
                    st.markdown(f'<div class="result-box">The estimated insurance premium is ₹ {prediction:.2f} (minimum default).</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box">The estimated insurance premium is ₹ {prediction:.2f}.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error occurred during prediction: {e}")

    # Static Built-By Section
    st.markdown("""
    <div class="about-box">
        <strong>Built with ❤️ by Hemanth</strong><br/>
        NIE Mysuru
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <hr />
        © 2025 MediQuote AI | All Rights Reserved
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
