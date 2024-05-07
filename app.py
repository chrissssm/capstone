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
st.header("Direct no-show calculation")
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

###############################################################
# Section for manually inputting booking variables and predicting show/no-show
st.header("Predict Show/No-Show for a Booking")

# Load the model
model = load_model()

# Define input fields for booking variables
no_of_adults = st.number_input("Number of Adults", min_value=1, step=1)
no_of_children = st.number_input("Number of Children", min_value=0, step=1)
no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, step=1)
no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, step=1)
type_of_meal_plan = st.selectbox("Type of Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
required_car_parking_space = st.number_input("Required Car Parking Space", min_value=0, step=1)
room_type_reserved = st.selectbox("Room Type Reserved", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
lead_time = st.number_input("Lead Time (Days)", min_value=0, step=1)
arrival_year = st.number_input("Arrival Year", min_value=2000, step=1)
arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, step=1)
arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, step=1)
market_segment_type = st.selectbox("Market Segment Type", ['Complementary', 'Online', 'Offline', 'Corporate', 'Aviation'])
repeated_guest = st.selectbox("Repeated Guest", [0, 1])
no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, step=1)
no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", min_value=0, step=1)
avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, step=0.01)
no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, step=1)

# Predict show/no-show for the input booking
if st.button("Predict Show/No-Show"):
    # Prepare input data for prediction
    input_data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }

    # Convert input data to DataFrame for consistency with model input
    input_df = pd.DataFrame([input_data])

    # Handle categorical features to match training
    input_df = pd.get_dummies(input_df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], drop_first=False)
    expected_categories = {
        'type_of_meal_plan': ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
        'room_type_reserved': ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'],
        'market_segment_type': ['Complementary', 'Online', 'Offline', 'Corporate', 'Aviation']
    }
    for column, categories in expected_categories.items():
        for category in categories:
            expected_col = f'{column}_{category}'
            if expected_col not in input_df.columns:
                input_df[expected_col] = 0

    # Reorder columns to match model input
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict show/no-show
    prediction = model.predict(input_df)

    # Display prediction result
    if prediction == 1:
        st.write("The booking is predicted to be a no-show.")
    else:
        st.write("The booking is predicted to be a show.")
