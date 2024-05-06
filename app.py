import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import july
from datetime import datetime

import plotly.express as px

# Dein Code zum Berechnen von `overbooked_rooms`

fig = px.imshow(
    overbooked_rooms.values,
    labels=dict(x="Date", y="Rooms"),
    x=overbooked_rooms.index,
    y=overbooked_rooms.columns,
    color_continuous_scale='Viridis'
)

st.plotly_chart(fig)





# Set app properties
st.set_page_config(
    page_title="Hotel Booking Optimizer",
    page_icon="🏨",
    layout="wide"
)

# Cache functions to load data and model
@st.cache(allow_output_mutation=True)
def load_model():
    filename = "model.sav"
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

# Function to adjust capacity and calculate overbooked rooms
def adjust_capacity(booking_data, max_capacity, show_rate):
    # Calculate available rooms based on max capacity and show rate
    available_rooms = int(max_capacity / show_rate)
    
    # Calculate overbooked rooms per date
    overbooked_rooms = available_rooms - booking_data.groupby('arrival_date').size()
    overbooked_rooms[overbooked_rooms < 0] = 0
    
    # Calculate additional revenue
    avg_room_price = booking_data['avg_price_per_room'].mean()
    additional_revenue = avg_room_price * overbooked_rooms.sum()
    
    return overbooked_rooms, additional_revenue

# Streamlit app
def main():
    

    # Section 1: Capacity Adjustment
    st.header("Capacity Adjustment")

    # Upload future bookings test set
    uploaded_file = st.file_uploader("Upload Future Bookings Test Set (CSV)", type=['csv'])

    if uploaded_file is not None:
        # Load booking data
        booking_data = pd.read_csv(uploaded_file)

        # Imaginary hotel max. capacity
        max_capacity = st.number_input("Enter Imaginary Hotel Max. Capacity", min_value=1)

        # Show rate (percentage of booked rooms that show up)
        show_rate = st.slider("Select Show Rate (%)", min_value=0, max_value=100, value=80, step=1)

        # Adjust capacity and calculate overbooked rooms
        overbooked_rooms, additional_revenue = adjust_capacity(booking_data, max_capacity, show_rate)

        # Plot calendar heatmap with adjusted capacity
        fig = px.imshow(overbooked_rooms.T, color_continuous_scale='Viridis')
        fig.update_layout(title="Adjusted Capacity Calendar Heatmap", xaxis_title="Date", yaxis_title="Rooms")
        st.plotly_chart(fig)

        # Show available rooms
        st.subheader("Available Rooms (Normal Max. Capacity):")
        st.write(max_capacity)

        # Show overbooked rooms per date
        st.subheader("Overbooked Rooms per Date:")
        st.write(overbooked_rooms)

        # Show additional revenue
        st.subheader("Additional Revenue:")
        st.write("$", round(additional_revenue, 2))

    # Section 2: Individual Booking Analysis
    st.header("Individual Booking Analysis")

    # Input booking details
    st.subheader("Input Booking Details:")
    # Here you can add input fields for each variable in the dataset

    # Button to predict no-show for individual booking
    if st.button("Predict No-Show"):
        # Perform prediction using the trained model (not shown here)
        # Display prediction result
        st.write("No-Show Prediction: Yes/No")

# Run the app
if __name__ == "__main__":
    main()
