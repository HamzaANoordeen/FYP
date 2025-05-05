import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Flight Ticket Price Dashboard", layout="wide")

@st.cache_data

def load_data():
    df = pd.read_csv("data/Clean_Dataset.csv")
    df.rename(columns={
        'price': 'ticket_price'
    }, inplace=True)
    return df

df = load_data()

st.title("✈️ Flight Ticket Price Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filter Options")
selected_airline = st.sidebar.multiselect("Select Airline(s)", df["airline"].unique(), default=df["airline"].unique())
selected_class = st.sidebar.radio("Select Class", df["class"].unique())
selected_days_range = st.sidebar.slider("Select Days Left Range", int(df["days_left"].min()), int(df["days_left"].max()), (10, 60))

# Filter data
filtered_df = df[
    (df["airline"].isin(selected_airline)) &
    (df["class"] == selected_class) &
    (df["days_left"].between(selected_days_range[0], selected_days_range[1]))
]

# Sample data for clearer plots
sampled_df = filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42)

st.subheader("💰 Ticket Price vs Days Left")
fig1 = px.scatter(
    sampled_df,
    x="days_left",
    y="ticket_price",
    color="airline",
    title="Ticket Price vs Days Left",
    template="plotly_white",
    height=400
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📉 Average Ticket Price by Class Over Days Left")
avg_price_class = df.groupby(["days_left", "class"])["ticket_price"].mean().reset_index()
fig2 = px.line(
    avg_price_class,
    x="days_left",
    y="ticket_price",
    color="class",
    title="Average Ticket Price by Class Over Days Left",
    template="plotly_white",
    height=400
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 Price Trend by Airline Over Days Left")
avg_price_airline = df.groupby(["days_left", "airline"])["ticket_price"].mean().reset_index()
fig3 = px.line(
    avg_price_airline,
    x="days_left",
    y="ticket_price",
    color="airline",
    title="Price Trend by Airline Over Days Left",
    template="plotly_white",
    height=400
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📉 Average Price Comparison by Class")
fig4 = px.line(
    df,
    x="days_left",
    y="ticket_price",
    color="class",
    title="Price Comparison Between Business and Economy Class",
    template="plotly_white",
    height=400
)
st.plotly_chart(fig4, use_container_width=True)

st.subheader("🛫 Number of Flights by Source City")
flight_counts = df["source_city"].value_counts().reset_index()
flight_counts.columns = ["City", "Flight Count"]
fig5 = px.bar(
    flight_counts,
    x="City",
    y="Flight Count",
    title="Number of Flights by Source City",
    template="plotly_white",
    color="Flight Count",
    height=400
)
st.plotly_chart(fig5, use_container_width=True)


st.subheader("💸 Cheapest Flight by Airline for Each Day Left")
cheapest_by_day = df.loc[df.groupby(["days_left", "airline"])['ticket_price'].idxmin()]
fig7 = px.line(
    cheapest_by_day,
    x="days_left",
    y="ticket_price",
    color="airline",
    title="Cheapest Flight by Airline Per Day Left",
    template="plotly_white",
    height=400
)
st.plotly_chart(fig7, use_container_width=True)

st.subheader("🧭 Stops Distribution")
fig8 = px.pie(
    df,
    names="stops",
    title="Distribution of Stops",
    template="plotly_white",
    hole=0.4
)
st.plotly_chart(fig8, use_container_width=True)

st.subheader("🛬 Destination Popularity")
dest_count = df["destination_city"].value_counts().reset_index()
dest_count.columns = ["Destination", "Count"]
fig9 = px.bar(
    dest_count,
    x="Destination",
    y="Count",
    title="Most Popular Destination Cities",
    color="Count",
    template="plotly_white",
    height=400
)
st.plotly_chart(fig9, use_container_width=True)

st.caption("Built for Flight Price Analysis 📊 | Visuals: Plotly | Interface: Streamlit")


# Footer
st.markdown("---")
st.markdown("Made by Hamza A Noordeen | Final Year Project 2025")

