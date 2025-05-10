import streamlit as st
import random
from datetime import datetime

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

        # Crop information form
        st.markdown("### 🌱 Add Your Crop Info")
        crop_name = st.selectbox("Select Crop", ["Onion", "Tomato", "Potato"])
        
        npk_n = st.number_input("N (Nitrogen) level", min_value=0, max_value=100, value=50)
        npk_p = st.number_input("P (Phosphorus) level", min_value=0, max_value=100, value=50)
        npk_k = st.number_input("K (Potassium) level", min_value=0, max_value=100, value=50)
        
        sowing_date = st.date_input("Sowing Date", min_value=datetime.today())
        crop_image = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png"])

        # Form submit
        if st.button("Submit Crop Info"):
            # Simulate health analysis based on NPK range
            health_status = "Good" if all(30 <= v <= 70 for v in [npk_n, npk_p, npk_k]) else "Needs Attention"
            price_prediction = random.randint(15, 120)  # Mock future price prediction
            st.success(f"Crop Health Status: **{health_status}**")
            st.success(f"Estimated Price for {crop_name}: ₹{price_prediction}/kg")

            # Crop profile details display
            st.markdown("### 📸 Crop Profile")
            st.image(crop_image) if crop_image else st.warning("No image uploaded")
            st.write(f"Crop Name: {crop_name}")
            st.write(f"NPK Levels → N: {npk_n}, P: {npk_p}, K: {npk_k}")
            st.write(f"Sowing Date: {sowing_date}")

            # Save to session (mock database storage)
            st.session_state.crop_data = {
                "crop_name": crop_name,
                "npk": {"N": npk_n, "P": npk_p, "K": npk_k},
                "sowing_date": sowing_date,
                "price_prediction": price_prediction
            }

    elif st.session_state.user_role == "Buyer":
        st.subheader("🏢 Buyer Dashboard")
        st.markdown("Coming soon: view farmer profiles, crop listings, and post demand.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.experimental_rerun()
