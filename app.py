import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

def set_background(image_path):
    base64_str = get_base64_image(image_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("background.jpg")

classification_model = joblib.load('classification_model.pkl')
regression_model = joblib.load('regression_model.pkl')
c_denier_encoder = joblib.load('c_denier_encoder.pkl')
c_capacity_encoder = joblib.load('c_capacity_encoder.pkl')
c_scaler = joblib.load('c_scaler.pkl')
c_X_train = joblib.load("c_X_train.pkl")
r_scaler = joblib.load('r_scaler.pkl')
r_X_train = joblib.load("r_X_train.pkl")

st.title('Nylon Dyeing Recipe Status Predictor')

def reset_prediction():
    st.session_state.prediction_class = None
    st.session_state.show_cost_section = False

recipe_quantity = st.number_input('Enter Recipe Quantity (kg):', min_value=0.001, step=0.001, format="%.3f", key="recipe_quantity", on_change=reset_prediction)
colour_shade = st.selectbox('Select the Colour Shade:', ['Very Light', 'Light', 'Medium', 'Dark', 'Very Dark'], key="colour_shade", on_change=reset_prediction)
first_colour = st.radio('Is the Color Being Created for the First Time in the Dye Lab?', ['Yes', 'No'], key="first_colour", on_change=reset_prediction)
colour_description = st.selectbox('Select the Colour Description:', ['Normal', 'Softner', 'Special Colour'], key="colour_description", on_change=reset_prediction)
lab_dip = st.radio('Is the Swatch Being Created in the Dye Lab?', ['Yes', 'No'], key="lab_dip", on_change=reset_prediction)
nylon_type = st.selectbox('Select the Nylon Type:', ['Stretch Nylon', 'Micro Fiber Streatch Nylon', 'Other'], key="nylon_type", on_change=reset_prediction)
denier = st.selectbox('Select the Denier Count:', [44, 70, 78, 100], key="denier", on_change=reset_prediction)
dyeing_method = st.selectbox('Select the Dyeing Method:', ['Bullet', 'Hank', 'Package'], key="dyeing_method", on_change=reset_prediction)
colour = st.selectbox('Select the Colour:', ['Black', 'White', 'Grey', 'Blue', 'Navy Blue', 'Green', 'Pink', 'Red',
                                 'Beige', 'Orange', 'Brown', 'Purple', 'Cream', 'Yellow', 'Maroon', 'Other'], key="colour", on_change=reset_prediction)
machine_capacity = st.selectbox('Select the Machine Capacity (Packages):', [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 24, 28, 30, 36, 42,
                                                                 25, 48, 54, 56, 75, 90, 104, 108, 132, 216, 264, 432, 558, 981],
                                key="machine_capacity", on_change=reset_prediction)

if 'prediction_class' not in st.session_state:
    st.session_state.prediction_class = None

if 'show_cost_section' not in st.session_state:
    st.session_state.show_cost_section = False

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

if st.session_state.prediction_class is not None:
    if st.session_state.prediction_class == 1:
        prediction_label = "RFT"
    else:
        prediction_label = "WFT. Please proceed with necessary steps."
    
    st.write(f"Prediction: {prediction_label}")

if st.session_state.show_cost_section:
    supplier = st.selectbox('Select the Supplier:', ['Rudolf', 'Ohyoung', 'Harris & Menuk'], key="supplier")
    iso_105 = st.radio('Does the dyestuff have high fastness ratings in ISO 105 series tests?', ['Yes', 'No'], key="iso_105")

    if st.button('Predict Cost'):
        cost_data = pd.DataFrame({
            'RecipeQty': recipe_quantity,
            'ColourShade': colour_shade,
            'ColourDescription': colour_description,
            'NylonType': nylon_type,
            'DyeingMethod': dyeing_method,
            'Supplier': supplier,
            'ISO105': iso_105
        }, index=[0])
        
        cost_dummy_cols = ['ColourShade', 'ColourDescription', 'NylonType', 'DyeingMethod', 'Supplier', 'ISO105']
        cost_dummies = pd.get_dummies(cost_data[cost_dummy_cols])
        cost_data = pd.concat([cost_data, cost_dummies], axis=1)
        cost_data = cost_data.drop(columns=cost_dummy_cols)
        
        missing_cols = [col for col in r_X_train if col not in cost_data.columns]
        for col in missing_cols:
            cost_data[col] = False
        
        cost_drop_first = ['ColourShade_Dark', 'ColourDescription_Normal', 'NylonType_Micro Fiber Streatch Nylon', 
                           'DyeingMethod_Bullet', 'Supplier_Harris & Menuk', 'ISO105_No']
        cost_drop = [col for col in cost_drop_first if col in cost_data.columns]
        cost_data = cost_data.drop(columns=cost_drop)
        cost_data = cost_data[r_X_train]
        cost_data['RecipeQty'] = r_scaler.transform(cost_data[['RecipeQty']])

        predicted_cost = regression_model.predict(cost_data)
        st.write(f"Predicted Cost: {predicted_cost[0]:.2f} LKR")

    if st.button('Cancel'):
        st.session_state.show_cost_section = False
        st.rerun()
