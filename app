# File: app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBClassifier

# --- Load the model and data ---
try:
    model = xgb_model  # The trained model must be in memory
    data = cleaned_data  # The cleaned dataset with country, continent, blood types, etc.
except NameError:
    st.error("Make sure to define 'xgb_model' and 'blood_data' in the notebook environment before running the app.")
    st.stop()

# Country-to-Continent mapping (ensure it's consistent with your model)
continent_coords = {
    'Africa': (1.6508, 10.2679),
    'Asia': (34.0479, 100.6197),
    'Europe': (54.5260, 15.2551),
    'North America': (54.5260, -105.2551),
    'South America': (-8.7832, -55.4915),
    'Oceania': (-25.2744, 133.7751)
}

continents = list(continent_coords.keys())
blood_types = ['A+', 'O+', 'B+', 'AB+', 'A-', 'B-', 'O-', 'AB-']

# Sidebar input
st.sidebar.header("Blood Type Matching Tool")
selected_blood = st.sidebar.selectbox("Select Your Blood Type", blood_types)
role = st.sidebar.radio("Are you a...", ['Donor', 'Recipient'])

# Prediction function (dummy implementation, replace with actual model prediction)
def get_predictions(blood_type, role):
    results = {}
    for continent in continents:
        input_df = pd.DataFrame({
            'Blood Type': [blood_type],
            'Role': [role],
            'Continent': [continent]
        })
        # Ensure input_df matches model's feature expectations
        pred = model.predict_proba(input_df)[0][1] * 100  # Assuming binary classification with probability output
        results[continent] = round(pred, 2)
    return results

# On submit
if st.sidebar.button("Submit"):
    tab_list = st.tabs(continents)
    predictions = get_predictions(selected_blood, role)

    for idx, continent in enumerate(continents):
        with tab_list[idx]:
            st.header(f"{continent} Overview for {selected_blood} ({role})")

            # Filter data for this continent
            continent_data = data[data['Continent'] == continent]

            if continent_data.empty:
                st.warning(f"No data available for {continent}.")
                continue

            # Map plot
            map_fig = px.scatter_geo(
                continent_data,
                locations="Country",
                locationmode="country names",
                hover_name="Country",
                color_discrete_sequence=['red'],
                scope="world",
                title=f"Countries in {continent}"
            )
            st.plotly_chart(map_fig)

            # Top 5 countries bar plot
            top_5 = continent_data[continent_data['Blood Type'] == selected_blood].nlargest(5, 'Count')
            bar_fig = px.bar(
                top_5,
                x='Country',
                y='Count',
                color='Country',
                title=f"Top 5 Countries with {selected_blood}"
            )
            st.plotly_chart(bar_fig)

            # Pie chart of all blood type distribution in this continent
            pie_data = continent_data.groupby('Blood Type').sum(numeric_only=True).reset_index()
            pie_fig = px.pie(
                pie_data,
                names='Blood Type',
                values='Count',
                title=f"Blood Type Distribution in {continent}"
            )
            st.plotly_chart(pie_fig)

            # Prediction output
            if role == 'Donor':
                st.success(f"Your blood type is needed by approximately **{predictions[continent]}%** of the population in {continent}.")
            else:
                st.info(f"You have a **{predictions[continent]}%** chance of finding a donor match in {continent}.")
