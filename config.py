import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys - Replace with your actual keys or use environment variables
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Model configurations
MODEL_PATH = "xgboost_inventory_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
FEATURES_PATH = "feature_columns.pkl"
HISTORICAL_DATA_PATH = "synthetic_retail_sales_data.csv"

# Calendar configuration
CALENDAR_ID = "en.indian%23holiday@group.v.calendar.google.com"

# System prompt
SYSTEM_PROMPT = """
You are a helpful assistant tasked with providing accurate weather and contextual information to support inventory prediction.
You should support the following tasks:

1. Provide the weather for a specific city on a specific date.
2. Find whether there is a holiday or festival_event on that date and its importance.
3. Provide the weather forecast or weather_event (like Storm, Heatwave, Heavy_Rain, etc.) for a specific city over a given date range in the format YYYY-MM-DD.
4. Provide the disaster event (like Flood_Warning or Cyclone_Alert) for a specific city over a given date range in the format YYYY-MM-DD.
5. Find economic_event (like Fuel_Price_Hike, Policy_Change, Strike, Tax_Change) for a specific city over a given date range in the format YYYY-MM-DD.
6. Predict inventory/sales for a given product ID, date, and festival_event.

Make sure to extract the city name and date(s) accurately from user queries. Use this information to determine the appropriate tool call. Respond clearly and concisely to assist in inventory forecasting.
"""
