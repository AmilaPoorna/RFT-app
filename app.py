import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

classification_model = joblib.load('classification_model.pkl')
regression_model = joblib.load('regression_model.pkl')
c_scaler = joblib.load('c_scaler.pkl')
c_denier_encoder = joblib.load('c_denier_encoder.pkl')
c_capacity_encoder = joblib.load('c_capacity_encoder.pkl')
c_X_train = joblib.load("c_X_train.pkl")
r_scaler = joblib.load('r_scaler.pkl')
r_X_train = joblib.load("r_X_train.pkl")

st.title('Nylon Dyeing Recipe Status Predictor')

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

if st.button('Predict'):

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
    st.write("Known Denier Classes:", c_denier_encoder.classes_)
    unknown_denier = set([denier]) - set(c_denier_encoder.classes_)
    st.write("Unseen Denier Labels:", unknown_denier)
    rft_data = pd.get_dummies(rft_data[['IsFirstColour', 'ColourShade', 'ColourDescription', 'IsLabDip', 'NylonType', 'DyeingMethod', 'Colour']])
    missing_cols = [col for col in c_X_train if col not in rft_data.columns]
    for col in missing_cols:
        rft_data[col] = False
    rft_data = rft_data[c_X_train]
    rft_data['RecipeQty'] = c_scaler.transform(rft_data[['RecipeQty']])
    rft_data['Denier'] = c_denier_encoder.transform(rft_data['Denier'])
    rft_data['MachineCapacity(Packages)'] = c_capacity_encoder.transform(rft_data['MachineCapacity(Packages)'])

    prediction_class = classification_model.predict(rft_data)

    if prediction_class[0] == 1:
        prediction_label = "RFT"
    elif prediction_class[0] == 0:
        prediction_label = "WFT"

    st.write(f"Prediction: {prediction_label}")

    if prediction_class[0] == 1:
        supplier = st.selectbox('Supplier', ['Rudolf', 'Ohyoung', 'Harris & Menuk'])
        iso_150 = st.radio('ISO 150', ['Yes', 'No'])

        if st.button('Predict Cost'):
            cost_data = pd.DataFrame({
                'RecipeQty': recipe_quantity,
                'ColourShade': colour_shade,
                'ColourDescription': colour_description,
                'NylonType': nylon_type,
                'DyeingMethod': dyeing_method,
                'Supplier': supplier,
                'ISO150': iso_150
            }, index=[0])
            cost_data = pd.get_dummies(rft_data[['ColourShade', 'ColourDescription', 'NylonType', 'DyeingMethod', 'Supplier', 'ISO105']])
            missing_cols = [col for col in r_X_train if col not in cost_data.columns]
            for col in missing_cols:
                cost_data[col] = False
            cost_data = cost_data[r_X_train]
            cost_data['RecipeQty'] = r_scaler.transform(cost_data[['RecipeQty']])

            predicted_cost = regression_model.predict(cost_data)
            st.write(f"Predicted Cost: {predicted_cost[0]:.2f} LKR")

    elif prediction_class[0] == 0:
        st.write("Please proceed with necessary steps.")

if st.button('Cancel'):
    st.experimental_rerun()
