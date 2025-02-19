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

st.title('Nylon Dyeing Recipe Status Predictor')

# Initialize session state variables
if 'prediction_class' not in st.session_state:
    st.session_state.prediction_class = None
    st.session_state.prediction_label = None
    st.session_state.show_value_fields = False
    st.session_state.prediction_value = None
    st.session_state.selected_supplier = "Rudolf"
    st.session_state.selected_iso150 = "Yes"
    st.session_state.previous_inputs = None  # Store previous input values

# User Inputs
recipe_quantity = st.number_input('Recipe Quantity (kg)', min_value=0.001, step=0.001)
colour_shade = st.selectbox('Colour Shade', ['Very Light', 'Light', 'Medium', 'Dark', 'Very Dark'])
first_colour = st.radio('First Colour', ['Yes', 'No'])
colour_description = st.selectbox('Colour Description', ['Normal', 'Softner', 'Special Colour'])
lab_dip = st.radio('Lab Dip', ['Yes', 'No'])
nylon_type = st.selectbox('Nylon Type', ['Stretch Nylon', 'Micro Fiber Streatch Nylon', 'Other'])
denier = st.selectbox('Denier', [44, 70, 78, 100])
dyeing_method = st.selectbox('Dyeing Method', ['Bullet', 'Hank', 'Package'])
colour = st.selectbox('Colour', ['Black', 'White', 'Grey', 'Blue', 'Navy Blue', 'Green', 'Pink', 'Red',
                                 'Beige', 'Orange', 'Brown', 'Purple', 'Cream', 'Yellow', 'Maroon', 'Other'])
machine_capacity = st.selectbox('Machine Capacity (Packages)', [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 24, 28, 30, 36, 42,
                                                                 25, 48, 54, 56, 75, 90, 104, 108, 132, 216, 264, 432, 558, 981])

# Create a dictionary to store current inputs
current_inputs = {
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

# Check if inputs have changed
if st.session_state.previous_inputs is not None and st.session_state.previous_inputs != current_inputs:
    st.session_state.prediction_value = None  # Reset predicted cost if any field changes

# Update previous inputs in session state
st.session_state.previous_inputs = current_inputs

# Predict Status Button
if st.button('Predict Status'):
    rft_data = pd.DataFrame(current_inputs, index=[0])
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
    rft_data = rft_data.drop(columns=rft_drop, errors='ignore')

    rft_data = rft_data[c_X_train]

    rft_data['Denier'] = c_denier_encoder.transform(rft_data['Denier'])
    rft_data['MachineCapacity(Packages)'] = c_capacity_encoder.transform(rft_data['MachineCapacity(Packages)'])
    rft_data['RecipeQty'] = c_scaler.transform(rft_data[['RecipeQty']])

    prediction_class = classification_model.predict(rft_data)

    st.session_state.prediction_class = prediction_class[0]
    st.session_state.prediction_label = "RFT" if prediction_class[0] == 1 else "WFT. Please proceed with necessary steps."
    st.session_state.show_value_fields = prediction_class[0] == 1
    st.session_state.prediction_value = None  # Reset cost when status is predicted

# Display Classification Result
if st.session_state.prediction_label is not None:
    st.write(f"Prediction: {st.session_state.prediction_label}")

# Cost Prediction for RFT class
if st.session_state.show_value_fields:
    supplier = st.selectbox('Supplier', ['Rudolf', 'Ohyoung', 'Harris & Menuk'],
                            index=['Rudolf', 'Ohyoung', 'Harris & Menuk'].index(st.session_state.selected_supplier))
    iso_150 = st.radio('ISO 150', ['Yes', 'No'], index=['Yes', 'No'].index(st.session_state.selected_iso150))

    # Update session state if supplier or ISO150 changes
    if supplier != st.session_state.selected_supplier or iso_150 != st.session_state.selected_iso150:
        st.session_state.selected_supplier = supplier
        st.session_state.selected_iso150 = iso_150

    if st.button('Predict Cost'):
        cost_data = pd.DataFrame({
            'RecipeQty': recipe_quantity,
            'ColourShade': colour_shade,
            'ColourDescription': colour_description,
            'NylonType': nylon_type,
            'DyeingMethod': dyeing_method,
            'Supplier': st.session_state.selected_supplier,
            'ISO150': st.session_state.selected_iso150
        }, index=[0])

        cost_dummy_cols = ['ColourShade', 'ColourDescription', 'NylonType', 'DyeingMethod', 'Supplier', 'ISO150']
        cost_dummies = pd.get_dummies(cost_data[cost_dummy_cols])
        cost_data = pd.concat([cost_data, cost_dummies], axis=1)
        cost_data = cost_data.drop(columns=cost_dummy_cols, errors='ignore')

        missing_cols = [col for col in r_X_train if col not in cost_data.columns]
        for col in missing_cols:
            cost_data[col] = False

        cost_data = cost_data[r_X_train]

        cost_data['RecipeQty'] = r_scaler.transform(cost_data[['RecipeQty']])

        st.session_state.prediction_value = regression_model.predict(cost_data)[0]  # Store predicted cost

    # Display predicted cost if available
    if st.session_state.prediction_value is not None:
        st.write(f"Predicted Cost: {st.session_state.prediction_value:.2f} LKR")

    # Cancel Button
    if st.button('Cancel'):
        st.session_state.show_value_fields = False
        st.session_state.prediction_value = None
        st.rerun()
