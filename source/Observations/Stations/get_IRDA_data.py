"""
File: get_IRDA_data.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: This is a description of what the script does
Created: 2024-03-21
"""

# Import statements
import pandas as pd
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

# Constants

# Functions

# Main execution ---------------------------------------
# load dataframe
df = pd.read_csv("C:\\Users\\sebastien.durocher\\PycharmProjects\\GetWeatherData\\data\\irda_stations\\CR1000_Station_Meteo_Table1.dat",na_values=['NAN'],skiprows=[0,2,3])

# Ensure that 'TIMESTAMP' is in datetime format
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# Get the latest timestamp
latest_timestamp = df['TIMESTAMP'].max()

# Get the current date and time
now = datetime.now()

# Calculate the difference in days
difference = now - latest_timestamp

# If the difference is more than three days, send an email
if difference > timedelta(days=3):
    msg = MIMEText("The latest observation is more than three days old.")
    msg['Subject'] = 'Error: Old Observation'
    msg['From'] = 'your_email@example.com'
    msg['To'] = 'recipient_email@example.com'

    # Send the email
    s = smtplib.SMTP('your_smtp_server.com')
    s.login('your_username', 'your_password')
    s.send_message(msg)
    s.quit()

if __name__ == "__main__":
    pass
