import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load models and encoders
classification_model = joblib.load('classification_model.pkl')
regression_model = joblib.load('regression_model.pkl')
c_denier_encoder = joblib.load('c_denier_encoder.pkl')
c_capacity_encoder = joblib.load('c_capacity_encoder.pkl')
c_scaler = joblib.load('c_scaler.pkl')
c_X_train = joblib.load("c_X_train.pkl")
r_scaler = joblib.load('r_scaler.pkl')
r_X_train = joblib.load("r_X_train.pkl")

# Set Page Configuration
st.set_page_config(page_title="Nylon Dyeing Predictor", page_icon="🎨", layout="wide")

# Custom CSS for Background
st.markdown(
    """
    <style>
    body {
        background-image: url('https://www.textileinsight.com/wp-content/uploads/2021/10/dyeing-process.jpg');
        background-size: cover;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
    }
    h1 {
        color: #1f4e79;
        text-align: center;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #163e5e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('🎨 Nylon Dyeing Recipe Status Predictor')

def reset_prediction():
    st.session_state.prediction_class = None
    st.session_state.show_cost_section = False

# User Inputs
recipe_quantity = st.number_input('Recipe Quantity (kg)', min_value=0.001, step=0.001, format="%.3f", key="recipe_quantity", on_change=reset_prediction)
colour_shade = st.selectbox('Colour Shade', ['Very Light', 'Light', 'Medium', 'Dark', 'Very Dark'], key="colour_shade", on_change=reset_prediction)
first_colour = st.radio('First Colour', ['Yes', 'No'], key="first_colour", on_change=reset_prediction)
colour_description = st.selectbox('Colour Description', ['Normal', 'Softner', 'Special Colour'], key="colour_description", on_change=reset_prediction)
lab_dip = st.radio('Lab Dip', ['Yes', 'No'], key="lab_dip", on_change=reset_prediction)
nylon_type = st.selectbox('Nylon Type', ['Stretch Nylon', 'Micro Fiber Streatch Nylon', 'Other'], key="nylon_type", on_change=reset_prediction)
denier = st.selectbox('Denier', [44, 70, 78, 100], key="denier", on_change=reset_prediction)
dyeing_method = st.selectbox('Dyeing Method', ['Bullet', 'Hank', 'Package'], key="dyeing_method", on_change=reset_prediction)
colour = st.selectbox('Colour', ['Black', 'White', 'Grey', 'Blue', 'Navy Blue', 'Green', 'Pink', 'Red',
                                 'Beige', 'Orange', 'Brown', 'Purple', 'Cream', 'Yellow', 'Maroon', 'Other'], key="colour", on_change=reset_prediction)
machine_capacity = st.selectbox('Machine Capacity (Packages)', [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 24, 28, 30, 36, 42,
                                                                 25, 48, 54, 56, 75, 90, 104, 108, 132, 216, 264, 432, 558, 981],
                                key="machine_capacity", on_change=reset_prediction)

# Session States
if 'prediction_class' not in st.session_state:
    st.session_state.prediction_class = None

if 'show_cost_section' not in st.session_state:
    st.session_state.show_cost_section = False

# Prediction Button
if st.button('Predict Status'):
    input_data = {
        'RecipeQty': recipe_quantity,
        'ColourShade': colour_shade,
        'IsFirstColour': first_colour,
        'ColourDescription': colour_description,
        'IsLabDip': lab_dip,
        'NylonType': nylon_type,
        'Denier': denier,
        'DyeingMethod': dyeing_method,
        'Colour': colour,
        'MachineCapacity(Packages)': machine_capacity
    }

    rft_data = pd.DataFrame(input_data, index=[0])
    rft_dummy_cols = ['IsFirstColour', 'ColourShade', 'ColourDescription', 'IsLabDip', 'NylonType', 'DyeingMethod', 'Colour']
    rft_dummies = pd.get_dummies(rft_data[rft_dummy_cols])
    rft_data = pd.concat([rft_data, rft_dummies], axis=1)
    rft_data = rft_data.drop(columns=rft_dummy_cols)
    
    missing_cols = [col for col in c_X_train if col not in rft_data.columns]
    for col in missing_cols:
        rft_data[col] = False

    rft_drop_first = ['IsFirstColour_No', 'ColourShade_Dark', 'ColourDescription_Normal', 'IsLabDip_No', 
                      'NylonType_Micro Fiber Streatch Nylon', 'DyeingMethod_Bullet', 'Colour_Beige']
    rft_drop = [col for col in rft_drop_first if col in rft_data.columns]
    rft_data = rft_data.drop(columns=rft_drop)
    rft_data = rft_data[c_X_train]
    rft_data['Denier'] = c_denier_encoder.transform(rft_data['Denier'])
    rft_data['MachineCapacity(Packages)'] = c_capacity_encoder.transform(rft_data['MachineCapacity(Packages)'])
    rft_data['RecipeQty'] = c_scaler.transform(rft_data[['RecipeQty']])
    
    prediction_class = classification_model.predict(rft_data)
    st.session_state.prediction_class = prediction_class[0]
    
    if prediction_class[0] == 1:
        st.session_state.show_cost_section = True
    else:
        st.session_state.show_cost_section = False

# Display Prediction
if st.session_state.prediction_class is not None:
    prediction_label = "✅ RFT" if st.session_state.prediction_class == 1 else "⚠️ WFT. Please proceed with necessary steps."
    st.subheader(f"Prediction: {prediction_label}")

# Cost Prediction Section
if st.session_state.show_cost_section:
    supplier = st.selectbox('Supplier', ['Rudolf', 'Ohyoung', 'Harris & Menuk'], key="supplier")
    iso_105 = st.radio('ISO 105', ['Yes', 'No'], key="iso_105")

    if st.button('Predict Cost'):
        cost_data = pd.DataFrame({'RecipeQty': recipe_quantity, 'ColourShade': colour_shade,
                                  'ColourDescription': colour_description, 'NylonType': nylon_type,
                                  'DyeingMethod': dyeing_method, 'Supplier': supplier, 'ISO105': iso_105}, index=[0])
        
        predicted_cost = regression_model.predict(pd.get_dummies(cost_data))
        st.subheader(f"💰 Predicted Cost: {predicted_cost[0]:,.2f} LKR")

    if st.button('Cancel'):
        st.session_state.show_cost_section = False
        st.rerun()
