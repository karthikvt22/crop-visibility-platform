import streamlit as st
import random
import pandas as pd

# --- Set up page ---
st.set_page_config(page_title="Crop Visibility Platform", layout="wide")

# --- Session state for login ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None

# --- App Title and Intro ---
st.title("🌾 Crop Visibility Platform")
st.markdown("""
This platform helps farmers and buyers connect by showing:
- 🌿 Crop health based on NPK, weather, and crop age  
- 📊 Market price analysis and predictions  
- 🧑‍🌾 Farmer crop profiles  
- 🏢 Demand postings from organizations
""")
st.divider()

# --- Login / Signup ---
if not st.session_state.logged_in:
    st.subheader("Login or Sign Up")

    role = st.radio("Who are you?", ["Farmer", "Buyer"], horizontal=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            # Mock login logic
            st.session_state.logged_in = True
            st.session_state.user_role = role
            st.success(f"Welcome, {username} 👋! Logged in as {role}")
            st.experimental_rerun()
        else:
            st.error("Please enter both username and password.")

# --- After login ---
else:
    if st.session_state.user_role == "Farmer":
        st.subheader("👨‍🌾 Farmer Dashboard")

        # Crop selection
        st.markdown("### 🧪 Crop Health Analysis")
        crop_name = st.selectbox("Select your crop", ["Onion", "Tomato", "Potato"])

        # Simulated NPK values
        npk_values = {
            "N": random.randint(10, 80),
            "P": random.randint(5, 40),
            "K": random.randint(20, 90)
        }
        st.write(f"NPK Levels → N: {npk_values['N']}, P: {npk_values['P']}, K: {npk_values['K']}")

        # Health evaluation
        health_status = "Good" if all(30 <= v <= 70 for v in npk_values.values()) else "Needs Attention"
        st.info(f"Crop Health Status: **{health_status}**")

        # Market prediction (mock)
        st.markdown("### 📈 Market Price Prediction")
        future_price = random.randint(15, 120)
        st.success(f"Estimated price in 2 months for {crop_name}: ₹{future_price}/kg")

        # Buyer interest mock
        st.markdown("### 💬 Buyer Interest")
        st.write("👤 **AgroMart Pvt Ltd** is interested in 500kg of Onion.")
        st.write("👤 **FreshBasket Org** wants to connect for Tomato supply.")

    elif st.session_state.user_role == "Buyer":
        st.subheader("🏢 Buyer Dashboard")
        st.markdown("Coming soon: view farmer profiles, crop listings, and post demand.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.experimental_rerun()
