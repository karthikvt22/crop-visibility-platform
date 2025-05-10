import streamlit as st

# Set up page
st.set_page_config(page_title="Crop Visibility Platform", layout="wide")

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None

# Title and intro
st.title("🌾 Crop Visibility Platform")
st.markdown("""
This platform helps farmers and buyers connect by showing:
- 🌿 Crop health based on NPK, weather, and crop age  
- 📊 Market price analysis and predictions  
- 🧑‍🌾 Farmer crop profiles  
- 🏢 Demand postings from organizations
""")
st.divider()

# Login / Signup
if not st.session_state.logged_in:
    st.subheader("Login or Sign Up")

    role = st.radio("Who are you?", ["Farmer", "Buyer"], horizontal=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            # Mock logic
            st.session_state.logged_in = True
            st.session_state.user_role = role
            st.success(f"Welcome, {username} 👋! Logged in as {role}")
            st.experimental_rerun()
        else:
            st.error("Please enter both username and password.")

else:
    # Show different dashboards
    if st.session_state.user_role == "Farmer":
        st.subheader("👨‍🌾 Farmer Dashboard")
        st.markdown("✔️ Market Overview\n✔️ Crop Health\n✔️ Price Prediction\n✔️ Buyer Interests")

    elif st.session_state.user_role == "Buyer":
        st.subheader("🏢 Buyer Dashboard")
        st.markdown("✔️ Top Performing Crops\n✔️ Farmer Listings\n✔️ Contact Farmers\n✔️ Post Demand")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.experimental_rerun()
