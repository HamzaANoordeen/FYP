import streamlit as st
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(page_title="Flight Booking Dashboard", layout="wide")

# Load data function
@st.cache_data
def load_data():
    clean_df = pd.read_csv(r"C:\Users\hamza\OneDrive\Desktop\FYP\Flight booking dashboard\Data\Clean_Dataset.csv")
    business_df = pd.read_csv(r"C:\Users\hamza\OneDrive\Desktop\FYP\Flight booking dashboard\Data\business.csv")
    economy_df = pd.read_csv(r"C:\Users\hamza\OneDrive\Desktop\FYP\Flight booking dashboard\Data\economy.csv")
    return clean_df, business_df, economy_df

clean_df, business_df, economy_df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")
sample_size = st.sidebar.slider("Sample Size (Clean Dataset)", 100, 5000, 1000)
sampled_df = clean_df.sample(n=sample_size, random_state=42)

# Title
st.title("✈️ Flight Booking Insights")
st.markdown("Analyze ticket prices across airlines, classes, stops, and booking windows.")

# Plot 1: Price vs Days Left
st.subheader("Price vs Days Before Traveling")
fig1 = px.scatter(sampled_df, x="days_left", y="price", color="class",
                  title="Ticket Price by Booking Window", trendline="ols")
st.plotly_chart(fig1, use_container_width=True)

# Plot 2: Average Price by Airline
st.subheader("Average Ticket Price by Airline")
avg_price = sampled_df.groupby("airline")["price"].mean().sort_values()
fig2 = px.bar(avg_price, x=avg_price.index, y=avg_price.values,
              labels={"x": "Airline", "y": "Average Price"},
              title="Average Price per Airline")
st.plotly_chart(fig2, use_container_width=True)

# Plot 3: Price vs Stops
st.subheader("Ticket Price by Number of Stops")
fig3 = px.box(sampled_df, x="stops", y="price", color="class",
              title="Price Variation by Stops")
st.plotly_chart(fig3, use_container_width=True)

# Plot 4: Class Share
st.subheader("Ticket Class Distribution")
class_counts = sampled_df["class"].value_counts()
fig4 = px.pie(names=class_counts.index, values=class_counts.values,
              title="Share of Ticket Classes")
st.plotly_chart(fig4, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("📊 Final Year Project | Built with Streamlit + Plotly")
