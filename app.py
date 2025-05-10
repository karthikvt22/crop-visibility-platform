import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheets Auth
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "C:/Users/Karthik/Desktop/crop-visibility-platform-2cd3eaba3ee3.json", scope)
client = gspread.authorize(creds)

# Open Login Data sheet
sheet = client.open("Login Data").sheet1  # Make sure sheet name is exactly "Login Data"

st.set_page_config(page_title="Crop Visibility Platform")

st.title("🌾 Crop Visibility Platform")
st.markdown("""
This platform helps farmers and buyers connect by showing:

🌿 Crop health based on NPK, weather, and crop age  
📊 Market price analysis and predictions  
🧑‍🌾 Farmer crop profiles  
🏢 Demand postings from organizations  
""")

st.header("Login or Sign Up")

role = st.radio("Who are you?", ["Farmer", "Buyer"])
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    login_success = False
    # Fetch and normalize records
    raw_records = sheet.get_all_records()
    records = []
    for row in raw_records:
        # strip whitespace from keys and values, and lowercase the keys
        normalized = {k.strip().lower(): str(v).strip() for k, v in row.items()}
        records.append(normalized)

    # normalize user input
    u = username.strip()
    p = password.strip()
    r = role.lower().strip()

    for row in records:
        if row.get('username') == u and row.get('password') == p and row.get('role') == r:
            login_success = True
            break

    if login_success:
        st.success(f"Welcome, {username} 👋! Logged in as {role}")
        # TODO: Show dashboard based on role
    else:
        st.error("Invalid credentials or role mismatch")

