import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and dataframe
df = pd.read_csv("./data.csv").iloc[:, 1:]
pipe = pickle.load(open("./lifeexpectancy.pkl", "rb"))

# Perform one-hot encoding on the 'Country' column
df_encoded = pd.get_dummies(df, columns=['Country'])

# Encode the "Status" column
df_encoded['Status'] = df_encoded['Status'].replace({'Developing': 1, 'Developed': 0})

st.title("Life Expectancy Predictor")

# User input
country = st.selectbox('Country', df['Country'].unique())
status = 1 if st.selectbox('Status', df['Status'].unique()) == 'Developing' else 0
adult_mortality = st.number_input('Adult Mortality')
infant_deaths = st.number_input('Infant Deaths')
alcohol = st.number_input('Alcohol')
percentage_expenditure = st.number_input('Percentage Expenditure')
hepatitis_b = st.number_input('Hepatitis B')
measles = st.number_input('Measles')
bmi = st.number_input('BMI')
under_five_deaths = st.number_input('Under-Five Deaths')
polio = st.number_input('Polio')
total_expenditure = st.number_input('Total Expenditure')
diphtheria = st.number_input('Diphtheria')
hiv_aids = st.number_input('HIV/AIDS')
gdp = st.number_input('GDP')
population = st.number_input('Population')
thinness_1_19_years = st.number_input('Thinness 1-19 Years')
thinness_5_9_years = st.number_input('Thinness 5-9 Years')
income_composition = st.number_input('Income Composition of Resources')
schooling = st.number_input('Schooling')

# Encode the user input for country
country_encoded = 'Country_' + country.replace(' ', '_')
if country_encoded in df_encoded.columns:
    query = pd.DataFrame([[0] * 202], columns=df_encoded.columns)
    # Fill other country columns with 0
    country_columns = [col for col in df_encoded.columns if col.startswith('Country_')]
    other_countries = [col for col in country_columns if col != country_encoded]
    query[other_countries] = 0
    # Set the encoded country as the last column
    query[country_encoded] = 1
else:
    # Handle the case when the country is not found in the original dataset
    st.error('Country not found in the dataset')

print(query)

# Prepare the remaining query data for prediction
query.loc[0, 'Status'] = 1 if status == 'Developing' else 0
query.loc[0, 'Adult Mortality'] = adult_mortality
query.loc[0, 'Infant Deaths'] = infant_deaths
query.loc[0, 'Alcohol Intake(L)'] = alcohol
query.loc[0, 'Percentage Expenditure'] = percentage_expenditure
query.loc[0, 'HepB Vaccination %'] = hepatitis_b
query.loc[0, 'Measles'] = measles
query.loc[0, 'BMI'] = bmi
query.loc[0, 'Under Five Deaths'] = under_five_deaths
query.loc[0, 'Pol3 Vaccination %'] = polio
query.loc[0, 'Total Expenditure'] = total_expenditure
query.loc[0, 'Diphtheria Vaccination %'] = diphtheria
query.loc[0, 'HIV/AIDS'] = hiv_aids
query.loc[0, 'GDP'] = gdp
query.loc[0, 'Population'] = population
query.loc[0, 'Thinness 10-19 years'] = thinness_1_19_years
query.loc[0, 'Thinness 5-9 years'] = thinness_5_9_years
query.loc[0, 'Resources Income Composition'] = income_composition
query.loc[0, 'Schooling'] = schooling

# Prediction
if st.button('Predict Life Expectancy'):
    print(query.columns)
    prediction = str(pipe.predict(query.values)[0]* 100)
    st.title("Predicted Life Expectancy: " + prediction + " year")
