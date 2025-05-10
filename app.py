import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Set up Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name("crop-visibility-platform-2cd3eaba3ee3.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open("User Credentials").sheet1

# Function to check login credentials
def check_login(username, password, role):
    # Check each row to see if the username, password, and role match
    for row in sheet.get_all_records():
        # Debugging: Output the row content
        print("Row content:", row)  # This line will print the entire row to the console

        # Safely strip whitespace and handle potential None values
        stored_username = row.get('username', '').strip() if row.get('username') else ''
        stored_password = row.get('password', '').strip() if row.get('password') else ''
        stored_role = row.get('role', '').strip() if row.get('role') else ''

        # Debugging: Output the processed credentials
        print(f"Checking: username='{stored_username}', password='{stored_password}', role='{stored_role}'")  # Debugging

        # Compare the provided data with stored data
        if stored_username == username and stored_password == password and stored_role == role:
            return True
    return False

# Streamlit login form
st.title("🌾 Crop Visibility Platform")

role = st.selectbox("Who are you?", ["Farmer", "Buyer"])
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if check_login(username, password, role):
        st.success(f"Welcome, {username} 👋! Logged in as {role}")
    else:
        st.error("Invalid credentials or role mismatch")
