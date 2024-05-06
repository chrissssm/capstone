import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import july
from datetime import datetime

import plotly.express as px




# Set app properties
st.set_page_config(
    page_title="Hotel Booking Optimizer",
    page_icon="🏨",
    layout="wide"
)

# Cache functions to load data and model
@st.cache(allow_output_mutation=True)
def load_model():
    filename = r"C:\hotel\model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model

# Define app header
st.title("Hotel Booking Optimizer")
st.markdown("🏨🛏️💸 Optimize your hotel room bookings and maximize revenue through strategic overbooking!")

# Section for uploading and processing booking data
st.header("Upload and Analyze Booking Data")
uploaded_file = st.file_uploader("Choose a booking data file (in CSV format)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['date'] = pd.to_datetime(data['full_date'])  # Assuming 'full_date' is in a proper date format
    st.write("Data successfully uploaded! Here's a snippet of the dataset:")
    st.dataframe(data.head())

    model = load_model()

    # Handle categorical features to match training
    expected_categories = {
        'market_segment_type': ['Complementary', 'Online', 'Offline', "Corporate", "Aviation"],
        'type_of_meal_plan': ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
        'room_type_reserved': ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']
    }
    data = pd.get_dummies(data, columns=['market_segment_type', 'type_of_meal_plan', 'room_type_reserved'], drop_first=False)
    for column, categories in expected_categories.items():
        for category in categories:
            expected_col = f'{column}_{category}'
            if expected_col not in data.columns:
                data[expected_col] = 0

    date_series = data['date']

    model_data = data.drop(columns=['date'], errors='ignore')
    model_columns = model.feature_names_in_
    model_data = model_data.reindex(columns=model_columns, fill_value=0)

    predictions = model.predict(model_data)
    data['is_no_show'] = predictions
    data['date'] = date_series

    summary_data = data.groupby('date')['is_no_show'].sum().reset_index()
    summary_data['potential_revenue'] = summary_data['is_no_show'] * data['avg_price_per_room'].mean()

    st.write("Predicted No-Shows (Potential Overbookings):")
    st.dataframe(summary_data.head())

    if not pd.api.types.is_datetime64_any_dtype(summary_data['date']):
        summary_data['date'] = pd.to_datetime(summary_data['date'])
    start_date = summary_data['date'].min()
    end_date = summary_data['date'].max()
    dates = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()

    if dates:
        values = summary_data.set_index('date').reindex(dates, fill_value=0)['is_no_show'].tolist()
        # Ensure a Matplotlib figure is correctly handled
        fig, ax = plt.subplots()
        july.heatmap(
            dates,
            values,
            ax=ax,
            title="Daily Predicted No-Shows",
            cmap="coolwarm",
            month_grid=True
        )
        st.pyplot(fig)
    else:
        st.error("No dates available to generate heatmap.")

    st.header("Detailed Revenue and Overbooking Analysis")
    period = st.selectbox("Choose the period for analysis:", ["Daily", "Weekly", "Monthly", "Yearly"])

    if period == "Daily":
        display_data = summary_data
    elif period == "Weekly":
        display_data = summary_data.set_index('date').resample('W').sum().reset_index()
    elif period == "Monthly":
        display_data = summary_data.set_index('date').resample('M').sum().reset_index()
    elif period == "Yearly":
        display_data = summary_data.set_index('date').resample('Y').sum().reset_index()

    st.write(f"Overbooking and Revenue Analysis ({period}):")
    st.dataframe(display_data[['date', 'is_no_show', 'potential_revenue']])

###################################################################################################

import base64

# Define the function before it's called
def get_binary_file_downloader_html(dataframe):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="hotel_data_with_is_no_show.csv">Download CSV File</a>'
    return href

# Section for uploading data to calculate 'is_no_show' and providing a CSV download
st.header("Calculate 'is_no_show' and Download Data")
uploaded_data = st.file_uploader("Upload data to calculate 'is_no_show' (in CSV format)", type="csv")
if uploaded_data is not None:
    data_to_calculate = pd.read_csv(uploaded_data)
    data_to_calculate['date'] = pd.to_datetime(data_to_calculate['full_date'])  # Assuming 'full_date' is in a proper date format
    st.write("Data successfully uploaded! Here's a snippet of the dataset:")
    st.dataframe(data_to_calculate.head())

    model = load_model()

    # Handle categorical features to match training
    expected_categories = {
        'market_segment_type': ['Complementary', 'Online', 'Offline', "Corporate", "Aviation"],
        'type_of_meal_plan': ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
        'room_type_reserved': ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']
    }
    data_to_calculate = pd.get_dummies(data_to_calculate, columns=['market_segment_type', 'type_of_meal_plan', 'room_type_reserved'], drop_first=False)
    for column, categories in expected_categories.items():
        for category in categories:
            expected_col = f'{column}_{category}'
            if expected_col not in data_to_calculate.columns:
                data_to_calculate[expected_col] = 0

    date_series_calculate = data_to_calculate['date']

    model_data_calculate = data_to_calculate.drop(columns=['date'], errors='ignore')
    model_columns_calculate = model.feature_names_in_
    model_data_calculate = model_data_calculate.reindex(columns=model_columns_calculate, fill_value=0)

    predictions_calculate = model.predict(model_data_calculate)
    data_to_calculate['is_no_show'] = predictions_calculate
    data_to_calculate['date'] = date_series_calculate

    st.write("Data with 'is_no_show' column calculated:")
    st.dataframe(data_to_calculate.head())

    # Download CSV button
    csv_download_link = get_binary_file_downloader_html(data_to_calculate)
    st.markdown(csv_download_link, unsafe_allow_html=True)


