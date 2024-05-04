import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import july
from datetime import datetime

# Set app properties
st.set_page_config(
    page_title="Hotel Booking Optimizer",
    page_icon="🏨",
    layout="wide"
)

# Cache functions to load data and model
@st.cache(allow_output_mutation=True)
def load_model():
    filename = "hotel_booking_model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model

# Define app header
st.title("Hotel Booking Optimizer")
st.markdown("🏨🛏️💸 Optimize your hotel room bookings and maximize revenue through strategic overbooking!")

# Section for uploading and processing booking data
st.header("Upload and Analyze Booking Data")
uploaded_file = st.file_uploader("Choose a booking data file (in CSV format)", type="csv")
if uploaded_file is not None:
    # Read and display the uploaded dataset
    data = pd.read_csv(uploaded_file)
    data['date'] = pd.to_datetime(data['arrival_year'].astype(str) + '-' + data['arrival_month'] + '-' + data['arrival_date'].astype(str))
    st.write("Data successfully uploaded!")
    st.dataframe(data.head())

    # Load model and make predictions
    model = load_model()
    features = data[['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
                     'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
                     'room_type_reserved', 'lead_time', 'market_segment_type', 'repeated_guest',
                     'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                     'avg_price_per_room', 'no_of_special_requests']]
    predictions = model.predict(features)
    data['is_no_show'] = (predictions == 'No-Show').astype(int)

    # Summary and visualization
    summary_data = data.groupby('date')['is_no_show'].sum().reset_index()
    summary_data['potential_revenue'] = summary_data['is_no_show'] * data['avg_price_per_room'].mean()

    st.write("Predicted No-Shows (Potential Overbookings):")
    st.dataframe(summary_data.head())

    # Heatmap visualization
    st.subheader("Daily Predicted No-Shows Heatmap")
    dates = july.utils.date_range(summary_data['date'].min(), summary_data['date'].max())
    values = summary_data.set_index('date').reindex(dates, fill_value=0)['is_no_show'].tolist()
    heatmap = july.heatmap(
        dates[0].year, dates[0].month,
        values,
        title="Daily Predicted No-Shows",
        cmap="coolwarm",
        month_grid=True
    )
    fig, ax = plt.subplots()
    ax.imshow(heatmap)
    ax.axis('off')
    st.pyplot(fig)

    # Interactive revenue and overbooking data view
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
