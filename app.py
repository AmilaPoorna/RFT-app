import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

classification_model = joblib.load('classification_model.pkl')
regression_model = joblib.load('regression_model.pkl')
c_denier_encoder = joblib.load('c_denier_encoder.pkl')
c_capacity_encoder = joblib.load('c_capacity_encoder.pkl')
c_scaler = joblib.load('c_scaler.pkl')
c_X_train = joblib.load("c_X_train.pkl")
r_scaler = joblib.load('r_scaler.pkl')
r_X_train = joblib.load("r_X_train.pkl")

st.title('Nylon Dyeing Recipe Status Predictor')

# Session state to manage cost prediction visibility
if "show_cost_section" not in st.session_state:
    st.session_state.show_cost_section = False
if "status_predicted" not in st.session_state:
    st.session_state.status_predicted = False

# Function to reset cost prediction section
def reset_cost_section():
    st.session_state.show_cost_section = False
    st.session_state.status_predicted = False

# Recipe Status Inputs
recipe_quantity = st.number_input('Recipe Quantity (kg)', min_value=0.001, step=0.001, format="%.3f", on_change=reset_cost_section)

colour_shade = st.selectbox('Colour Shade', ['Very Light', 'Light', 'Medium', 'Dark', 'Very Dark'], on_change=reset_cost_section)

first_colour = st.radio('First Colour', ['Yes', 'No'], on_change=reset_cost_section)

colour_description = st.selectbox('Colour Description', ['Normal', 'Softner', 'Special Colour'], on_change=reset_cost_section)

lab_dip = st.radio('Lab Dip', ['Yes', 'No'], on_change=reset_cost_section)

nylon_type = st.selectbox('Nylon Type', ['Stretch Nylon', 'Micro Fiber Streatch Nylon', 'Other'], on_change=reset_cost_section)

denier = st.selectbox('Denier', [44, 70, 78, 100], on_change=reset_cost_section)

dyeing_method = st.selectbox('Dyeing Method', ['Bullet', 'Hank', 'Package'], on_change=reset_cost_section)

colour = st.selectbox('Colour', ['Black', 'White', 'Grey', 'Blue', 'Navy Blue', 'Green', 'Pink', 'Red',
                                           'Beige', 'Orange', 'Brown', 'Purple', 'Cream', 'Yellow', 'Maroon', 'Other'], on_change=reset_cost_section)

machine_capacity = st.selectbox('Machine Capacity (Packages)', [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 24, 28, 30, 36, 42,
                                                               25, 48, 54, 56, 75, 90, 104, 108, 132, 216, 264, 432, 558, 981], on_change=reset_cost_section)

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
    rft_drop_first = ['IsFirstColour_No', 'ColourShade_Dark', 'ColourDescription_Normal', 'IsLabDip_No', 'NylonType_Micro Fiber Streatch Nylon', 'DyeingMethod_Bullet', 'Colour_Beige']
    rft_drop = [col for col in rft_drop_first if col in rft_data.columns]
    rft_data = rft_data.drop(columns=rft_drop)
    rft_data = rft_data[c_X_train]
    rft_data['Denier'] = c_denier_encoder.transform(rft_data['Denier'])
    rft_data['MachineCapacity(Packages)'] = c_capacity_encoder.transform(rft_data['MachineCapacity(Packages)'])
    rft_data['RecipeQty'] = c_scaler.transform(rft_data[['RecipeQty']])
    
    prediction_class = classification_model.predict(rft_data)

    if prediction_class[0] == 1:
        prediction_label = "RFT"
    else:
        prediction_label = "WFT"
    
    st.write(f"Prediction: {prediction_label}")

    if prediction_class[0] == 0:
        st.write("Please proceed with necessary steps.")

    if prediction_class[0] == 1:
        st.session_state.status_predicted = True

# Cost Prediction Section - only show if status was predicted as RFT
if st.session_state.status_predicted:
    supplier = st.selectbox('Supplier', ['Rudolf', 'Ohyoung', 'Harris & Menuk'])
    iso_150 = st.radio('ISO 150', ['Yes', 'No'])

    if st.button('Predict Cost'):
        st.session_state.show_cost_section = True

# Show Cost Section only if activated
if st.session_state.show_cost_section:
    cost_data = pd.DataFrame({
        'RecipeQty': recipe_quantity,
        'ColourShade': colour_shade,
        'ColourDescription': colour_description,
        'NylonType': nylon_type,
        'DyeingMethod': dyeing_method,
        'Supplier': supplier,
        'ISO150': iso_150
    }, index=[0])
    
    cost_dummy_cols = ['ColourShade', 'ColourDescription', 'NylonType', 'DyeingMethod', 'Supplier', 'ISO150']
    cost_dummies = pd.get_dummies(cost_data[cost_dummy_cols])
    cost_data = pd.concat([cost_data, cost_dummies], axis=1)
    cost_data = cost_data.drop(columns=cost_dummy_cols)
    missing_cols = [col for col in c_X_train if col not in cost_data.columns]
    for col in missing_cols:
        cost_data[col] = False
    cost_drop_first = ['ColourShade_Dark', 'ColourDescription_Normal', 'NylonType_Micro Fiber Streatch Nylon', 'DyeingMethod_Bullet', 'Supplier_Harris & Menuk', 'ISO150_No']
    cost_drop = [col for col in cost_drop_first if col in cost_data.columns]
    cost_data = cost_data.drop(columns=cost_drop)
    st.write("Processed Cost Data:", cost_data)
    cost_data = cost_data[r_X_train]
    cost_data['RecipeQty'] = r_scaler.transform(cost_data[['RecipeQty']])

    predicted_cost = regression_model.predict(cost_data)
    st.write(f"Predicted Cost: {predicted_cost[0]:.2f} LKR")

    if st.button('Cancel'):
        st.session_state.show_cost_section = False
        st.experimental_rerun()
