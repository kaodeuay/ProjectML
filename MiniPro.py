import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn
st.write(""" 

## Forest Fires

""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')

# -- Define function to display widgets and store data


def get_input():
    # Display widgets and store their values in variables
    v_X = st.sidebar.radio('X', ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    v_Y = st.sidebar.radio('Y', ['2', '3', '4', '5', '6', '7', '8', '9'])
    v_month = st.sidebar.radio('month', ['February','March','April','June','July','August','September','October','December'])
    v_day = st.sidebar.radio('Day', ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    v_FFMC = st.sidebar.slider('FFMC', 18.7, 96.2, 0.1)
    v_DMC = st.sidebar.slider('DMC', 1.1, 291.3, 0.1)
    v_DC = st.sidebar.slider('DC', 7.9, 860.6, 0.1)
    v_ISI = st.sidebar.slider('ISI', 0.0, 56.1, 0.1)
    v_temp = st.sidebar.slider('temp', 2.2, 33.3, 0.1)
    v_RH = st.sidebar.slider('RH', 15, 100, 1)
    v_wind = st.sidebar.slider('wind', 0.4, 9.4, 0.1)
    v_rain = st.sidebar.slider('rain', 0.0, 6.4, 0.1)

    # Month
    if v_month == 'February':
        v_month = '2'
    elif v_month == 'March':
        v_month = '3'
    elif v_month == 'April':
        v_month = '4'
    elif v_month == 'June':
        v_month = '6'
    elif v_month == 'July':
        v_month = '7'
    elif v_month == 'August':
        v_month = '8'
    elif v_month == 'September':
        v_month = '9'
    elif v_month == 'October':
        v_month = '10'
    elif v_month == 'December':
        v_month = '12'
    # Day
    if v_day == 'Monday':
        v_day = '1'
    elif v_day == 'Tuesday':
        v_day = '2'
    elif v_day == 'Wednesday':
        v_day = '3'
    elif v_day == 'Thursday':
        v_day = '4'
    elif v_day == 'Friday':
        v_day = '5'
    elif v_day == 'Saturday':
        v_day = '6'
    elif v_day == 'Sunday':
        v_day = '7'

    # Store user input data in a dictionary
    data = {'X': v_X,
            'Y': v_Y,
            'month': v_month,
            'day': v_day,
            'FFMC': v_FFMC,
            'DMC': v_DMC,
            'DC': v_DC,
            'ISI': v_ISI,
            'temp': v_temp,
            'RH': v_RH,
            'wind': v_wind,
            'rain': v_rain,}

    # Create a data frame from the above dictionary
    data_df = pd.DataFrame(data, index=[0])
    return data_df

# -- Call function to display widgets and get data from user
df = get_input()

st.header('Application of Status Prediction:')

# -- Display new data from user inputs:
st.subheader('User Input:')
st.write(df)

# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('ML_A.csv')
df = pd.concat([df, data_sample],axis=0)

###Data Cleaning & Feature Engineering###
#drop
df = df.drop(columns=['area'])
df = df.drop(columns=['Unnamed: 0'])
df_num=df
# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(df_num)
# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization_ML1.pkl', 'rb'))
#Apply the normalization model to new data
x_new = load_nor.transform(df)
x_new = x_new[:1]
st.subheader('Normalization Input:')
st.write(x_new)
# -- Reads the saved classification model
load_LR = pickle.load(open('LR_ML1.pkl', 'rb'))
# Apply model for prediction
prediction = load_LR.predict(x_new)
prediction = prediction[:1]
st.subheader('Prediction:')
st.write(prediction)

#[theme]
#primaryColor="#aee1e1"
#backgroundColor="#d3e0dc"
#secondaryBackgroundColor="#fcd1d1"
#textColor="#5c5c5c"
#font="monospace"
