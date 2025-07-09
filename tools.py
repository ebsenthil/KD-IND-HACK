from langchain.tools import tool
import requests
from datetime import datetime, timedelta
from tavily import TavilyClient
from inventory_predictor import InventoryPredictor
from config import TAVILY_API_KEY, GOOGLE_API_KEY, CALENDAR_ID, MODEL_PATH, ENCODERS_PATH, FEATURES_PATH, HISTORICAL_DATA_PATH

# Initialize Tavily client
client = TavilyClient(api_key=TAVILY_API_KEY)

# Initialize inventory predictor
predictor = InventoryPredictor(
    model_path=MODEL_PATH,
    encoders_path=ENCODERS_PATH,
    features_path=FEATURES_PATH,
    historical_data_path=HISTORICAL_DATA_PATH
)

@tool
def get_holidays_on_date(date: str) -> str:
    """
    Fetches public holidays or festival_event in India for a specific date (yyyy-mm-dd).

    Args:
        date (str): Date in 'YYYY-MM-DD' format.

    Returns:
        str: Holiday name(s) which is festival_event or message if none found.
    """
    try:
        # Parse date and prepare RFC3339 format
        dt = datetime.strptime(date, "%Y-%m-%d")
        time_min = dt.strftime("%Y-%m-%dT00:00:00Z")
        time_max = dt.strftime("%Y-%m-%dT23:59:59Z")

        url = (
            f"https://www.googleapis.com/calendar/v3/calendars/{CALENDAR_ID}/events"
            f"?key={GOOGLE_API_KEY}&timeMin={time_min}&timeMax={time_max}"
            f"&singleEvents=true&orderBy=startTime"
        )

        response = requests.get(url)
        data = response.json()
        events = data.get("items", [])

        if not events:
            return f"No public holidays on {date} in India."

        result = f"Holidays on {date} in India:\n"
        for event in events:
            result += f"- {event['summary']}\n"

        return result.strip()

    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_weather_forecast(city: str, start: str, end: str) -> str:
    """
    Get weather forecast for a city between start and end dates (YYYY-MM-DD).
    Falls back if the date is beyond forecast range.
    """
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    today = datetime.today()

    if (start_date - today).days > 15:
        return (
            f"Sorry, I can only fetch forecasts up to 15 days ahead. "
            f"{start} is too far in the future. Would you like historical averages instead?"
        )

    # Example for Chennai (for simplicity)
    latitude = 13.08
    longitude = 80.27

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=auto&start_date={start}&end_date={end}"
    )

    response = requests.get(url)
    data = response.json()

    if 'daily' not in data:
        return "No weather data available for the specified range."

    # Format the weather output nicely
    daily = data['daily']
    forecast = ""
    for i in range(len(daily['time'])):
        forecast += (
            f"{daily['time'][i]}:\n"
            f"- Max Temp: {daily['temperature_2m_max'][i]}°C\n"
            f"- Min Temp: {daily['temperature_2m_min'][i]}°C\n"
            f"- Rainfall: {daily['precipitation_sum'][i]} mm\n\n"
        )

    return forecast.strip()

@tool
def search_economic_events(city: str) -> str:
    """
    Search economic events like Policy_Change,Fuel_Price_Hike,Strike,Tax_Change.
    Returns a summary string.
    """
    query = f"Search economic events like Policy_Change,Fuel_Price_Hike,Strike,Tax_Change etc in {city}"
    try:
        response = client.search(query=query, search_depth="advanced", include_answer=True)
        return response.get("answer", "No summary available.")
    except Exception as e:
        return f"Error during search: {e}"

@tool
def search_disaster_events(city: str) -> str:
    """
    Search natural disaster such as Flood warnning, cycle alert etc,
    Returns a summary string.
    """
    query = f"Search natural disaster such as Flood warnning, cycle alert etc in {city}"
    try:
        response = client.search(query=query, search_depth="advanced", include_answer=True)
        return response.get("answer", "No summary available.")
    except Exception as e:
        return f"Error during search: {e}"

@tool
def search_weather_events(city: str) -> str:
    """
    Search weather events like Heavy_rain, Heatwave, storm etc
    Returns a summary string.
    """
    query = f"Search weather events like Heavy_rain, Heatwave, storm etc in {city}"
    try:
        response = client.search(query=query, search_depth="advanced", include_answer=True)
        return response.get("answer", "No summary available.")
    except Exception as e:
        return f"Error during search: {e}"

@tool
def predict_inventory_enriched(
    product_id: str,
    date: str,
    festival_event: str = "",
    economic_event: str = "",
    natural_disaster: str = "",
    weather_event: str = "",
) -> str:
    """
    Predict inventory/sales for a given product using enriched event metadata.

    Args:
        product_id (str): Product SKU.
        date (str): Date in YYYY-MM-DD.
        festival_event (str): Name of the festival.
        economic_event (str): Economic trigger/event.
        natural_disaster (str): Disaster type like Flood, Cyclone.
        weather_event (str): Weather condition (Rain, Heatwave, etc.).

    Returns:
        str: Predicted sales.
    """
    result = predictor.predict_simple(
        product_id=product_id,
        date=date,
        festival_event=festival_event,
        economic_event=economic_event,
        natural_disaster=natural_disaster,
        weather_event=weather_event,
    )

    return (
        f"Predicted Sales for {product_id} on {date} during {festival_event or 'this period'}: "
        f"{result['predicted_sales']} units"
    )

# List of all tools
tools = [
    get_weather_forecast,
    get_holidays_on_date,
    search_economic_events,
    search_disaster_events,
    search_weather_events,
    predict_inventory_enriched
]
