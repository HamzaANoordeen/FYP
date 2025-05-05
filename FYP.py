import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="Optimal Flight Booking Time", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Clean_Dataset.csv")
    df = df.dropna(subset=["ticket price", "days before booking and traveling"])
    df["days before booking and traveling"] = pd.to_numeric(df["days before booking and traveling"], errors='coerce')
    df["ticket price"] = pd.to_numeric(df["ticket price"], errors='coerce')
    return df.dropna(subset=["ticket price", "days before booking and traveling"])

df = load_data()

# Title
st.title("✈️ Optimal Flight Booking Time Explorer")
st.markdown("""
Explore how flight ticket prices change based on when you book. Filter by class, airline, and route to discover optimal booking windows.
""")

# Sidebar filters
st.sidebar.header("Filter Flights")

class_option = st.sidebar.selectbox("Select Flight Class", df["class"].dropna().unique())
airlines = st.sidebar.multiselect("Choose Airlines", df["airline name"].dropna().unique(), default=df["airline name"].unique())
city_options = df[["departure city", "destination city"]].dropna()

source = st.sidebar.selectbox("Departure City", city_options["departure city"].unique())
destination = st.sidebar.selectbox("Destination City", city_options["destination city"].unique())
days_range = st.sidebar.slider("Days Before Departure", 0, int(df["days before booking and traveling"].max()), (10, 60))

# Filter data
filtered_df = df[
    (df["class"] == class_option) &
    (df["airline name"].isin(airlines)) &
    (df["departure city"] == source) &
    (df["destination city"] == destination) &
    (df["days before booking and traveling"].between(days_range[0], days_range[1]))
]

# Show stats
st.subheader("📊 Summary Statistics")
st.write(filtered_df[["ticket price", "days before booking and traveling"]].describe())

# Scatter Plot
st.subheader("💰 Ticket Price vs Days Before Booking")
fig = px.scatter(filtered_df, x="days before booking and traveling", y="ticket price",
                 color="airline name", trendline="lowess",
                 labels={"days before booking and traveling": "Days Before Travel", "ticket price": "Ticket Price (₦)"},
                 title="Price Trend by Days Before Departure")
st.plotly_chart(fig, use_container_width=True)

# Box Plot
st.subheader("📦 Price Distribution")
box_fig = px.box(filtered_df, x="airline name", y="ticket price", color="airline name",
                 labels={"ticket price": "Ticket Price (₦)", "airline name": "Airline"},
                 title="Ticket Price Distribution by Airline")
st.plotly_chart(box_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built for final year project - Flight Booking Optimization 🛫")
