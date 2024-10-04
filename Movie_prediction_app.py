import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('movie_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the feature data (X) to get unique values for dropdowns
X = pd.read_csv('movie_features.csv')

# Load the original dataframe
df = pd.read_csv('movie_data_full.csv')

# Define categorical columns after renaming
categorical_cols_X = ['Genres', 'Country', 'Lead Actor', 'Director']

# Define the Streamlit app
st.title("Movie Success Predictor")

# Store label encoding mappings for categorical features
label_encoding_mappings = {}

# **Define budget_duration_mappings dictionary**
budget_duration_mappings = {}
for feature in ['Budget', 'Duration of the Movie']:
    original_values = df[feature.replace('Duration of the Movie', 'duration').replace('Budget', 'budget')]
    scaled_values = X[feature].unique()
    budget_duration_mappings[feature] = dict(zip(scaled_values, original_values))  # Store mapping of scaled to original values

# Create input widgets for each feature
input_features = {}
for feature in X.columns:
    if feature in categorical_cols_X:
        # Store original and encoded values for mapping
        unique_values = X[feature].unique()
        original_values = df[
            feature.replace('Lead Actor', 'actor_1_name')
            .replace('Director', 'director_name')
            .replace('Genres', 'genres')
            .replace('Country', 'country')
        ].unique()  # Get original values from 'df'
        label_encoding_mappings[feature] = dict(zip(unique_values, original_values))

        # Display dropdown with original values
        input_features[feature] = st.selectbox(feature, list(original_values))
    elif feature == 'Duration of the Movie':
        # Duration dropdown from 1 to 5 hours
        input_features[feature] = st.selectbox(feature, np.arange(1, 6, 0.5))  # Hours in steps of 0.5
    elif feature == 'Budget':
        # Budget input with plus button (step of 1 million)
        min_budget = df['budget'].min()
        max_budget = df['budget'].max()
        input_features[feature] = st.number_input(
            feature,
            min_value=min_budget,
            max_value=max_budget,
            value=min_budget,
            step=1000000.0,  # Changed step to float64
        )
    else:
        # For numerical features, use the original code
        unique_values = sorted(X[feature].unique())
        input_features[feature] = st.selectbox(feature, unique_values)

# Create a predict button
if st.button("Predict"):
    # Create input data for prediction
    input_data = pd.DataFrame([input_features])

    # Map categorical values back to encoded values
    for feature in categorical_cols_X:
        input_data[feature] = input_data[feature].map(
            lambda x: [k for k, v in label_encoding_mappings[feature].items() if v == x][0]
            if x in label_encoding_mappings[feature].values()
            else x
        )

    # Map Budget and Duration back to scaled values (if needed)
    for feature in ['Budget', 'Duration of the Movie']:
        if feature in budget_duration_mappings:  # budget_duration_mappings definition missing
            input_data[feature] = input_data[feature].map(
                lambda x: [k for k, v in budget_duration_mappings.items() if v == x][0]
                if x in budget_duration_mappings.values()
                else x
            )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction result
    if prediction == 0:
        st.write("Prediction: Flop Movie")
    elif prediction == 1:
        st.write("Prediction: Average Movie")
    else:
        st.write("Prediction: Hit Movie")