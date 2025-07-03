
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(page_title="Flight Ticket Price Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/Clean_Dataset.csv")
    df.rename(columns={'price': 'ticket_price'}, inplace=True)
    return df

df = load_data()

@st.cache_resource
def load_model(class_type):
    file_name = "economy_model.pkl" if class_type.lower() == "economy" else "business_model.pkl"
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features'], data['scaler']

def predict_flight_price(input_data, class_type):
    model, features, scaler = load_model(class_type)
    input_dict = {col: 0 for col in features}
    input_dict['duration_mins'] = input_data['duration_mins']
    input_dict['stop'] = input_data['stop']
    input_dict['day_of_week'] = input_data['day_of_week']
    input_dict['month'] = input_data['month']
    input_dict['is_weekend'] = 1 if input_data['day_of_week'] in [5, 6] else 0
    airline_col = f"airline_{input_data['airline']}"
    if airline_col in features:
        input_dict[airline_col] = 1
    input_df = pd.DataFrame([input_dict])
    num_cols = ['duration_mins', 'stop', 'day_of_week', 'month']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    return model.predict(input_df[features])[0]

st.title("✈️ Flight Ticket Price Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filter Options")
selected_airline = st.sidebar.multiselect("Select Airline(s)", df["airline"].unique(), default=df["airline"].unique())
selected_class = st.sidebar.radio("Select Class", df["class"].unique())
selected_days_range = st.sidebar.slider("Select Days Left Range", int(df["days_left"].min()), int(df["days_left"].max()), (10, 60))

filtered_df = df[
    (df["airline"].isin(selected_airline)) &
    (df["class"] == selected_class) &
    (df["days_left"].between(selected_days_range[0], selected_days_range[1]))
]

sampled_df = filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42)

st.subheader("💰 Ticket Price vs Days Left")
fig1 = px.scatter(sampled_df, x="days_left", y="ticket_price", color="airline", title="Ticket Price vs Days Left", template="plotly_white", height=400)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📉 Average Ticket Price by Class Over Days Left")
avg_price_class = df.groupby(["days_left", "class"])["ticket_price"].mean().reset_index()
fig2 = px.line(avg_price_class, x="days_left", y="ticket_price", color="class", title="Average Ticket Price by Class Over Days Left", template="plotly_white", height=400)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 Price Trend by Airline Over Days Left")
avg_price_airline = df.groupby(["days_left", "airline"])["ticket_price"].mean().reset_index()
fig3 = px.line(avg_price_airline, x="days_left", y="ticket_price", color="airline", title="Price Trend by Airline Over Days Left", template="plotly_white", height=400)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📉 Average Price Comparison by Class")
fig4 = px.line(df, x="days_left", y="ticket_price", color="class", title="Price Comparison Between Business and Economy Class", template="plotly_white", height=400)
st.plotly_chart(fig4, use_container_width=True)

st.subheader("🛫 Number of Flights by Source City")
flight_counts = df["source_city"].value_counts().reset_index()
flight_counts.columns = ["City", "Flight Count"]
fig5 = px.bar(flight_counts, x="City", y="Flight Count", title="Number of Flights by Source City", template="plotly_white", color="Flight Count", height=400)
st.plotly_chart(fig5, use_container_width=True)

st.subheader("💸 Cheapest Flight by Airline for Each Day Left")
cheapest_by_day = df.loc[df.groupby(["days_left", "airline"])['ticket_price'].idxmin()]
fig7 = px.line(cheapest_by_day, x="days_left", y="ticket_price", color="airline", title="Cheapest Flight by Airline Per Day Left", template="plotly_white", height=400)
st.plotly_chart(fig7, use_container_width=True)

st.subheader("🧭 Stops Distribution")
fig8 = px.pie(df, names="stops", title="Distribution of Stops", template="plotly_white", hole=0.4)
st.plotly_chart(fig8, use_container_width=True)

st.subheader("🛬 Destination Popularity")
dest_count = df["destination_city"].value_counts().reset_index()
dest_count.columns = ["Destination", "Count"]
fig9 = px.bar(dest_count, x="Destination", y="Count", title="Most Popular Destination Cities", color="Count", template="plotly_white", height=400)
st.plotly_chart(fig9, use_container_width=True)

st.subheader("🔮 Predict Flight Ticket Price")
with st.form("prediction_form"):
    airline = st.selectbox("Airline", ["Air India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"])
    travel_class = st.selectbox("Class", ["Economy", "Business"])
    departure_city = st.selectbox("Departure City", sorted(df["source_city"].unique()))
    arrival_city = st.selectbox("Arrival City", sorted(df["destination_city"].unique()))
    duration = st.number_input("Duration (in minutes)", min_value=30, max_value=1800, value=120)
    stops = st.selectbox("Number of Stops", [0, 1, 2])
    day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)), index=3)
    month = st.selectbox("Month of Travel", list(range(1, 13)), index=5)

      submitted = st.form_submit_button("Predict Price")
    if submitted:
        user_input = {
            "airline": airline,
            "class": travel_class,
            "departure_city": departure_city,
            "arrival_city": arrival_city,
            "duration_mins": duration,
            "stop": stops,
            "day_of_week": day_of_week,
            "month": month
        }
        
        # Only use the relevant fields for prediction
        prediction = predict_flight_price(user_input, travel_class)
        st.success(f"Predicted Ticket Price: ₹{prediction:,.2f}")


st.caption("Built for Flight Price Analysis 📊 | Visuals: Plotly | Interface: Streamlit")
st.markdown("---")
st.markdown("Made by Hamza A Noordeen | Final Year Project 2025")
