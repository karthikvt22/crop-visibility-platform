import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load credentials from JSON key file
creds = ServiceAccountCredentials.from_json_keyfile_name("crop-visibility-platform-2cd3eaba3ee3.json", scope)

# Authorize the client
client = gspread.authorize(creds)

# Open the login data sheet
login_sheet = client.open("Login Data").sheet1

# Open the farmer data sheet
farmer_sheet = client.open("Farmer Data").sheet1

# Open the buyer data sheet
buyer_sheet = client.open("Buyer Data").sheet1

def register_user(username, password, role):
    users = login_sheet.get_all_records()
    for user in users:
        if user["username"] == username:
            return False  # Username already exists
    login_sheet.append_row([username, role, password])
    return True

def login_user(username, password, role):
    users = login_sheet.get_all_records()
    for user in users:
        if user["username"] == username and user["password"] == password and user["role"] == role:
            return True
    return False
