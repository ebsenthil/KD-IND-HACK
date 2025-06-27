import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

fake = Faker()
np.random.seed(42)
random.seed(42)

# Define product categories and their characteristics
products = [
    {'product_id': f'SKU{str(i+1).zfill(3)}', 
     'product_category': random.choice(['Electronics', 'Clothing', 'Food', 'Home', 'Beauty', 
                                      'Sports', 'Books', 'Toys', 'Medicine', 'Stationery']),
     'base_demand': random.randint(50, 300),
     'price': round(random.uniform(10, 500), 2),
     'seasonality_factor': random.uniform(0.8, 1.2)}
    for i in range(50)
]

# Define external events for the year
festivals = [
    {'date': '2024-01-26', 'event': 'Republic_Day', 'impact': 'Medium'},
    {'date': '2024-03-08', 'event': 'Holi', 'impact': 'High'},
    {'date': '2024-08-15', 'event': 'Independence_Day', 'impact': 'Medium'},
    {'date': '2024-10-02', 'event': 'Gandhi_Jayanti', 'impact': 'Low'},
    {'date': '2024-11-12', 'event': 'Diwali', 'impact': 'High'},
    {'date': '2024-12-25', 'event': 'Christmas', 'impact': 'High'},
]

weather_disasters = [
    {'date': '2024-02-15', 'weather': 'Heavy_Rain', 'disaster': 'Flood_Warning', 'severity': 'High'},
    {'date': '2024-04-22', 'weather': 'Heatwave', 'disaster': 'None', 'severity': 'Medium'},
    {'date': '2024-06-10', 'weather': 'Storm', 'disaster': 'Cyclone_Alert', 'severity': 'High'},
    {'date': '2024-07-18', 'weather': 'Heavy_Rain', 'disaster': 'None', 'severity': 'Medium'},
    {'date': '2024-09-05', 'weather': 'Storm', 'disaster': 'None', 'severity': 'Low'},
]

economic_events = [
    {'date': '2024-01-15', 'event': 'Policy_Change', 'impact': 6},
    {'date': '2024-03-20', 'event': 'Fuel_Price_Hike', 'impact': 7},
    {'date': '2024-08-10', 'event': 'Strike', 'impact': 8},
    {'date': '2024-11-01', 'event': 'Tax_Change', 'impact': 5},
]

def get_external_events(date_str):
    """Get external events for a specific date"""
    events = {
        'weather_event': 'None',
        'weather_intensity': 'None',
        'natural_disaster': 'None',
        'disaster_severity': 0,
        'festival_event': 'None',
        'festival_impact_level': 'None',
        'economic_event': 'None',
        'economic_impact_score': 0,
        'combined_event_count': 0,
        'is_pre_event': 0,
        'is_post_event': 0
    }
    
    # Check festivals
    for festival in festivals:
        if festival['date'] == date_str:
            events['festival_event'] = festival['event']
            events['festival_impact_level'] = festival['impact']
            events['combined_event_count'] += 1
        # Pre/post event logic
        festival_date = datetime.strptime(festival['date'], '%Y-%m-%d')
        current_date = datetime.strptime(date_str, '%Y-%m-%d')
        if (current_date - festival_date).days == -1:
            events['is_pre_event'] = 1
        elif (current_date - festival_date).days == 1:
            events['is_post_event'] = 1
    
    # Check weather/disasters
    for weather in weather_disasters:
        if weather['date'] == date_str:
            events['weather_event'] = weather['weather']
            events['weather_intensity'] = weather['severity']
            events['natural_disaster'] = weather['disaster']
            events['disaster_severity'] = {'Low': 3, 'Medium': 6, 'High': 9}[weather['severity']]
            events['combined_event_count'] += 1
    
    # Check economic events
    for econ in economic_events:
        if econ['date'] == date_str:
            events['economic_event'] = econ['event']
            events['economic_impact_score'] = econ['impact']
            events['combined_event_count'] += 1
    
    return events

def calculate_sales_impact(product, date, day_of_week, month, events):
    """Calculate sales based on product characteristics and external factors"""
    base = product['base_demand']
    
    # Seasonal adjustment
    seasonal_multiplier = 1.0
    if month in [11, 12]:  # Festival season
        seasonal_multiplier = 1.3
    elif month in [6, 7, 8]:  # Monsoon
        seasonal_multiplier = 0.9
    
    # Day of week effect
    weekday_multiplier = 1.2 if day_of_week in [5, 6] else 1.0  # Weekend boost
    
    # External events impact
    event_multiplier = 1.0
    
    # Festival impact
    if events['festival_event'] != 'None':
        if product['product_category'] in ['Food', 'Clothing', 'Electronics']:
            event_multiplier *= {'Low': 1.2, 'Medium': 1.5, 'High': 2.0}[events['festival_impact_level']]
    
    # Weather impact
    if events['weather_event'] != 'None':
        if product['product_category'] == 'Medicine' and events['weather_event'] in ['Heavy_Rain', 'Storm']:
            event_multiplier *= 1.4
        elif product['product_category'] == 'Food' and events['natural_disaster'] != 'None':
            event_multiplier *= 1.8  # Panic buying
    
    # Pre/post event effects
    if events['is_pre_event']:
        event_multiplier *= 1.3  # Pre-event buying
    elif events['is_post_event']:
        event_multiplier *= 0.7  # Post-event drop
    
    # Calculate final sales with some randomness
    final_sales = base * seasonal_multiplier * weekday_multiplier * event_multiplier
    final_sales *= random.uniform(0.8, 1.2)  # Add noise
    
    return max(0, int(final_sales))

# Generate the dataset
data = []
start_date = datetime(2024, 1, 1)

for day in range(365):  # One year
    current_date = start_date + timedelta(days=day)
    date_str = current_date.strftime('%Y-%m-%d')
    day_of_week = current_date.weekday()
    month = current_date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    quarter = (month - 1) // 3 + 1
    
    # Get external events for this date
    events = get_external_events(date_str)
    
    for product in products:
        # Calculate sales for this product on this day
        sales_qty = calculate_sales_impact(product, current_date, day_of_week, month, events)
        
        # Price variations (occasional discounts)
        current_price = product['price']
        price_change_pct = 0
        if random.random() < 0.1:  # 10% chance of price change
            price_change_pct = random.uniform(-0.2, 0.1)  # -20% to +10%
            current_price *= (1 + price_change_pct)
        
        row = {
            # Sales table columns
            'date': date_str,
            'product_id': product['product_id'],
            'product_category': product['product_category'],
            'sales_qty': sales_qty,
            'price': round(current_price, 2),
            'store_id': 'STORE_001',
            'day_of_week': day_of_week,
            'month': month,
            'quarter': quarter,
            'is_weekend': is_weekend,
            'is_holiday': 1 if events['festival_event'] != 'None' else 0,
            'base_demand': product['base_demand'],
            'price_change_pct': round(price_change_pct * 100, 2),
            
            # External events columns
            'weather_event': events['weather_event'],
            'weather_intensity': events['weather_intensity'],
            'natural_disaster': events['natural_disaster'],
            'disaster_severity': events['disaster_severity'],
            'festival_event': events['festival_event'],
            'festival_impact_level': events['festival_impact_level'],
            'economic_event': events['economic_event'],
            'economic_impact_score': events['economic_impact_score'],
            'combined_event_count': events['combined_event_count'],
            'is_pre_event': events['is_pre_event'],
            'is_post_event': events['is_post_event']
        }
        
        data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Add lag features (simplified - using previous week's data)
df = df.sort_values(['product_id', 'date']).reset_index(drop=True)

lag_features = []
for product_id in df['product_id'].unique():
    product_data = df[df['product_id'] == product_id].copy()
    product_data['sales_lag_7d'] = product_data['sales_qty'].shift(7)
    product_data['sales_lag_30d'] = product_data['sales_qty'].shift(30)
    product_data['rolling_avg_7d'] = product_data['sales_qty'].rolling(window=7, min_periods=1).mean()
    product_data['rolling_avg_30d'] = product_data['sales_qty'].rolling(window=30, min_periods=1).mean()
    lag_features.append(product_data)

df_final = pd.concat(lag_features).sort_values(['date', 'product_id']).reset_index(drop=True)

# Fill NaN values
df_final['sales_lag_7d'] = df_final['sales_lag_7d'].fillna(df_final['base_demand'])
df_final['sales_lag_30d'] = df_final['sales_lag_30d'].fillna(df_final['base_demand'])

# Display basic statistics
print(f"Dataset created successfully!")
print(f"Total rows: {len(df_final)}")
print(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")
print(f"Products: {df_final['product_id'].nunique()}")
print(f"Product categories: {df_final['product_category'].unique()}")
print("\nSample data:")
print(df_final.head())

print("\nExternal events summary:")
print(f"Festival events: {df_final[df_final['festival_event'] != 'None']['festival_event'].value_counts()}")
print(f"Weather events: {df_final[df_final['weather_event'] != 'None']['weather_event'].value_counts()}")

# Save to CSV
df_final.to_csv('synthetic_retail_sales_data.csv', index=False)
print(f"\nData saved to 'synthetic_retail_sales_data.csv'")
print(f"Columns: {list(df_final.columns)}")
