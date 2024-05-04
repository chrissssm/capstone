#### Overall setup 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px

# Defining some general properties of the app
st.set_page_config(
    page_title= "Hotel Booking Optimizer",
    page_icon = "🏨",
    layout="wide"
    )

# Define Load functions
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    return(data.dropna())

@st.cache_resource
def load_model():
    filename = "model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model)

# Load Data and Model
data = load_data()
model = load_model()

st.title("Hotel Booking Optimizer")
st.markdown("🏨🛏️💸 This application can be used to determine how many hotel rooms can be overbooked 🏨🛏️💸")

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
