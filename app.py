import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Flight Ticket Price Dashboard", layout="wide")

@st.cache_data

def load_data():
    df = pd.read_csv("Clean_Dataset.csv")
    df.rename(columns={
        'price': 'ticket_price'
    }, inplace=True)
    return df

df = load_data()
import pickle

@st.cache_resource
def predict_flight_price(input_data):
    """Completely handles all feature name mismatches"""
    # 1. Load model and metadata
    with open('flight_price_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        expected_features = saved_data['features']
    
    # 2. Create a dictionary with ALL expected features initialized to 0
    features = {col: 0 for col in expected_features}
    
    # 3. Handle special cases
    # Convert duration if provided
    if 'duration' in input_data:
        duration = str(input_data['duration'])
        hours = int(duration.split('h')[0]) if 'h' in duration else 0
        mins = int(duration.split('m')[0].split('h')[-1]) if 'm' in duration else 0
        features['duration_mins'] = hours * 60 + mins
    
    # Set numerical features
    if 'days_left' in input_data:
        features['days_left'] = input_data['days_left']
    
    # 4. Automatically match categorical features
    categorical_fields = ['airline', 'source_city', 'departure_time', 
                        'arrival_time', 'destination_city', 'class', 'stops']
    
    for field in categorical_fields:
        if field in input_data:
            # Find matching feature (case and space insensitive)
            search_term = f"{field}_{input_data[field]}".lower().replace(' ', '_')
            matching_feature = next(
                (f for f in expected_features 
                 if f.lower().replace(' ', '_') == search_term),
                None
            )
            if matching_feature:
                features[matching_feature] = 1
    
    # 5. Create DataFrame with EXACT feature order
    input_df = pd.DataFrame([features])[expected_features]
    
    return model.predict(input_df)[0]

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

st.subheader("🔮 Predict Flight Ticket Price")


with st.form("prediction_form"):
    airline = st.selectbox("Airline", ["Air India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"])
    destination_city = st.selectbox("Destination City", ["Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
    flight_class = st.selectbox("Class", ["Economy", "Business"])
    days_left = st.number_input("Days Before Departure", min_value=1, max_value=365)
    
    if st.form_submit_button("Predict Price"):
        input_data = {
            "airline": airline,
            "destination_city": destination_city,
            "class": flight_class,
            "days_left": days_left
        }
        # Predict (using the prediction function from earlier)
        prediction = predict_flight_price(input_data)
        st.success(f"Predicted Price: ₹{prediction:,.2f}")

st.caption("Built for Flight Price Analysis 📊 | Visuals: Plotly | Interface: Streamlit")


# Footer
st.markdown("---")
st.markdown("Made by Hamza A Noordeen | Final Year Project 2025")

