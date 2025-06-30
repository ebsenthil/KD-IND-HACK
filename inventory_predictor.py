# Enhanced Inventory Prediction System with Minimal Inputs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
print("Current working directory:", os.getcwd())


class InventoryPredictor:
    def __init__(self, model_path='inventory_prediction/xgboost_inventory_model.pkl', 
                 encoders_path='inventory_prediction/label_encoders.pkl', 
                 features_path='inventory_prediction/feature_columns.pkl',
                 historical_data_path='inventory_prediction/synthetic_retail_sales_data.csv'):
        """
        Initialize the inventory predictor
        """
        # Load the trained model and preprocessors
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)
        self.feature_columns = joblib.load(features_path)
        
        # Load historical data for calculating defaults and lags
        self.historical_data = pd.read_csv(historical_data_path)
        self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
        
        # Create product lookup tables
        self._create_product_lookups()
        
        print("Inventory Predictor initialized successfully!")
    
    def _create_product_lookups(self):
        """Create lookup tables for product defaults"""
        # Product information lookup
        self.product_info = self.historical_data.groupby('product_id').agg({
            'product_category': 'first',
            'price': 'mean',
            'base_demand': 'mean',
            'sales_qty': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        self.product_info.columns = ['category', 'avg_price', 'avg_base_demand', 'avg_sales', 'sales_std']
        
        # Category defaults (for unknown products)
        self.category_defaults = self.historical_data.groupby('product_category').agg({
            'price': 'mean',
            'base_demand': 'mean',
            'sales_qty': 'mean'
        }).round(2)
        
        # Overall defaults
        self.overall_defaults = {
            'price': self.historical_data['price'].mean(),
            'base_demand': self.historical_data['base_demand'].mean(),
            'category': 'Electronics'  # Most common category
        }
    
    def _get_product_defaults(self, product_id):
        """Get default values for a product"""
        if product_id in self.product_info.index:
            return self.product_info.loc[product_id].to_dict()
        else:
            # Return overall defaults for unknown products
            return {
                'category': self.overall_defaults['category'],
                'avg_price': self.overall_defaults['price'],
                'avg_base_demand': self.overall_defaults['base_demand'],
                'avg_sales': self.overall_defaults['base_demand'],
                'sales_std': self.overall_defaults['base_demand'] * 0.2
            }
    
    def _calculate_date_features(self, date):
        """Calculate all date-related features"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        features = {
            'day_of_week': date.weekday(),  # 0=Monday, 6=Sunday
            'month': date.month,
            'quarter': date.quarter,
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'day_of_month': date.day,
            'week_of_year': date.isocalendar()[1]
        }
        
        # Simple holiday detection (you can enhance this)
        holidays = [
            (1, 1),   # New Year
            (1, 26),  # Republic Day
            (8, 15),  # Independence Day
            (10, 2),  # Gandhi Jayanti
            (12, 25), # Christmas
        ]
        features['is_holiday'] = 1 if (date.month, date.day) in holidays else 0
        
        return features
    
    def _calculate_lag_features(self, product_id, prediction_date):
        """Calculate lag and rolling average features"""
        if isinstance(prediction_date, str):
            prediction_date = pd.to_datetime(prediction_date)
        
        # Get historical data for this product
        product_data = self.historical_data[
            (self.historical_data['product_id'] == product_id) & 
            (self.historical_data['date'] < prediction_date)
        ].sort_values('date')
        
        if len(product_data) == 0:
            # No historical data - use product defaults
            defaults = self._get_product_defaults(product_id)
            return {
                'sales_lag_7d': defaults['avg_sales'],
                'sales_lag_30d': defaults['avg_sales'],
                'rolling_avg_7d': defaults['avg_sales'],
                'rolling_avg_30d': defaults['avg_sales']
            }
        
        # Calculate lag features
        recent_sales = product_data['sales_qty'].values
        
        # If we have enough data, use actual lags
        if len(recent_sales) >= 30:
            sales_lag_7d = recent_sales[-7]
            sales_lag_30d = recent_sales[-30]
            rolling_avg_7d = np.mean(recent_sales[-7:])
            rolling_avg_30d = np.mean(recent_sales[-30:])
        elif len(recent_sales) >= 7:
            sales_lag_7d = recent_sales[-min(7, len(recent_sales))]
            sales_lag_30d = recent_sales[-1]
            rolling_avg_7d = np.mean(recent_sales[-7:])
            rolling_avg_30d = np.mean(recent_sales)
        else:
            # Use available data
            sales_lag_7d = recent_sales[-1] if len(recent_sales) > 0 else defaults['avg_sales']
            sales_lag_30d = recent_sales[-1] if len(recent_sales) > 0 else defaults['avg_sales']
            rolling_avg_7d = np.mean(recent_sales) if len(recent_sales) > 0 else defaults['avg_sales']
            rolling_avg_30d = np.mean(recent_sales) if len(recent_sales) > 0 else defaults['avg_sales']
        
        return {
            'sales_lag_7d': round(sales_lag_7d, 2),
            'sales_lag_30d': round(sales_lag_30d, 2),
            'rolling_avg_7d': round(rolling_avg_7d, 2),
            'rolling_avg_30d': round(rolling_avg_30d, 2)
        }
    
    def _calculate_event_features(self, weather_event, natural_disaster, festival_event, economic_event):
        """Calculate event-related features"""
        # Event intensity mappings
        weather_intensity_map = {
            'None': 'None',
            'Heavy_Rain': 'High',
            'Extreme_Heat': 'High',
            'Storm': 'High',
            'Light_Rain': 'Low',
            'Cloudy': 'Low',
            'Sunny': 'None'
        }
        
        festival_impact_map = {
            'None': 'None',
            'Diwali': 'High',
            'Holi': 'High',
            'Christmas': 'High',
            'Eid': 'High',
            'New_Year': 'Medium',
            'Valentine': 'Low'
        }
        
        # Calculate external intensity score
        external_intensity = 0
        
        # Weather impact
        weather_intensity = weather_intensity_map.get(weather_event, 'Low')
        if weather_event != 'None':
            external_intensity += {'Low': 2, 'Medium': 4, 'High': 6}.get(weather_intensity, 0)
        
        # Festival impact
        festival_impact_level = festival_impact_map.get(festival_event, 'Low')
        if festival_event != 'None':
            external_intensity += {'Low': 3, 'Medium': 5, 'High': 8}.get(festival_impact_level, 0)
        
        # Natural disaster impact
        disaster_severity = 0
        if natural_disaster != 'None':
            disaster_severity = {'Flood': 7, 'Earthquake': 8, 'Cyclone': 6}.get(natural_disaster, 5)
            external_intensity += disaster_severity
        
        # Economic event impact
        economic_impact_score = 0
        if economic_event != 'None':
            economic_impact_score = {'Recession': -5, 'Boom': 5, 'Policy_Change': 3}.get(economic_event, 0)
            external_intensity += abs(economic_impact_score)
        
        # Count events
        events = [weather_event, natural_disaster, festival_event, economic_event]
        combined_event_count = sum(1 for event in events if event != 'None')
        
        # Pre/post event flags (simplified)
        is_pre_event = 1 if festival_event != 'None' else 0
        is_post_event = 0  # Would require date logic for actual implementation
        
        return {
            'weather_intensity': weather_intensity,
            'festival_impact_level': festival_impact_level,
            'disaster_severity': disaster_severity,
            'economic_impact_score': economic_impact_score,
            'external_intensity': external_intensity,
            'combined_event_count': combined_event_count,
            'is_pre_event': is_pre_event,
            'is_post_event': is_post_event
        }
    
    def predict_simple(self, product_id, date, weather_event='None', 
                      natural_disaster='None', festival_event='None', 
                      economic_event='None', custom_price=None, 
                      custom_base_demand=None):
        """
        Simplified prediction function requiring minimal inputs
        
        Parameters:
        - product_id: Product ID (string)
        - date: Prediction date (string or datetime)
        - weather_event: Weather condition (default: 'None')
        - natural_disaster: Natural disaster (default: 'None') 
        - festival_event: Festival/holiday (default: 'None')
        - economic_event: Economic condition (default: 'None')
        - custom_price: Override default price (optional)
        - custom_base_demand: Override default base demand (optional)
        
        Returns:
        - Dictionary with prediction results
        """
        
        # Get product defaults
        product_defaults = self._get_product_defaults(product_id)
        
        # Use custom values or defaults
        price = custom_price if custom_price is not None else product_defaults['avg_price']
        base_demand = custom_base_demand if custom_base_demand is not None else product_defaults['avg_base_demand']
        product_category = product_defaults['category']
        
        # Calculate all features automatically
        date_features = self._calculate_date_features(date)
        lag_features = self._calculate_lag_features(product_id, date)
        event_features = self._calculate_event_features(weather_event, natural_disaster, 
                                                       festival_event, economic_event)
        
        # Create complete input data
        input_data = {
            'product_id': product_id,
            'product_category': product_category,
            'price': price,
            'base_demand': base_demand,
            'weather_event': weather_event,
            'natural_disaster': natural_disaster,
            'festival_event': festival_event,
            'economic_event': economic_event,
            'price_change_pct': 0,  # Assuming no price change
            **date_features,
            **lag_features,
            **event_features
        }
        
        # Make prediction using the existing function logic
        prediction = self._predict_with_input_data(input_data)
        
        # Return comprehensive results
        return {
            'predicted_sales': prediction,
            'base_demand': base_demand,
            'uplift_units': prediction - base_demand,
            'uplift_percentage': round(((prediction - base_demand) / base_demand * 100), 1),
            'input_summary': {
                'product_id': product_id,
                'date': str(date),
                'weather_event': weather_event,
                'natural_disaster': natural_disaster,
                'festival_event': festival_event,
                'economic_event': economic_event,
                'calculated_price': price,
                'calculated_base_demand': base_demand
            },
            'feature_summary': {
                'external_intensity': event_features['external_intensity'],
                'is_weekend': date_features['is_weekend'],
                'is_holiday': date_features['is_holiday'],
                'combined_events': event_features['combined_event_count']
            }
        }
    
    def _predict_with_input_data(self, input_data):
        """Internal prediction function"""
        # Create DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Apply label encoding
        for col, encoder in self.label_encoders.items():
            if col in df_input.columns:
                try:
                    df_input[col + '_encoded'] = encoder.transform(df_input[col].astype(str))
                except:
                    # Handle unseen categories
                    df_input[col + '_encoded'] = 0
        
        # Select required features and handle missing ones
        feature_data = {}
        for feature in self.feature_columns:
            if feature in df_input.columns:
                feature_data[feature] = df_input[feature].iloc[0]
            else:
                feature_data[feature] = 0  # Default value for missing features
        
        # Create feature array
        X_input = pd.DataFrame([feature_data])
        
        # Make prediction
        prediction = self.model.predict(X_input)[0]
        
        return max(0, int(prediction))

# Usage Example and Testing Functions
def demo_simple_predictions():
    """Demonstrate the simplified prediction system"""
    
    # Initialize predictor (make sure model files exist)
    try:
        predictor = InventoryPredictor()
        print("✅ Predictor initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        print("Make sure model files exist in the current directory")
        return
    
    # Example predictions with minimal inputs
    test_cases = [
        {
            'name': 'Normal Day',
            'product_id': 'SKU001',
            'date': '2024-07-15'
        },
        {
            'name': 'Festival Sale',
            'product_id': 'SKU001', 
            'date': '2024-11-12',
            'festival_event': 'Diwali'
        },
        {
            'name': 'Weather Impact',
            'product_id': 'SKU001',
            'date': '2024-08-20',
            'weather_event': 'Heavy_Rain'
        },
        {
            'name': 'Multiple Events',
            'product_id': 'SKU001',
            'date': '2024-12-25',
            'festival_event': 'Christmas',
            'weather_event': 'Storm',
            'economic_event': 'Boom'
        }
    ]
    
    print("🔮 INVENTORY PREDICTIONS")
    print("=" * 50)
    
    for case in test_cases:
        print(f"\n📊 {case['name']}:")
        print("-" * 30)
        
        try:
            result = predictor.predict_simple(**case)
            
            print(f"Product: {result['input_summary']['product_id']}")
            print(f"Date: {result['input_summary']['date']}")
            print(f"Base Demand: {result['base_demand']} units")
            print(f"Predicted Sales: {result['predicted_sales']} units")
            print(f"Uplift: {result['uplift_units']} units ({result['uplift_percentage']}%)")
            
            # Show active events
            events = []
            if case.get('weather_event', 'None') != 'None':
                events.append(f"Weather: {case['weather_event']}")
            if case.get('festival_event', 'None') != 'None':
                events.append(f"Festival: {case['festival_event']}")
            if case.get('natural_disaster', 'None') != 'None':
                events.append(f"Disaster: {case['natural_disaster']}")
            if case.get('economic_event', 'None') != 'None':
                events.append(f"Economic: {case['economic_event']}")
            
            if events:
                print(f"Active Events: {', '.join(events)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

# Function to create a batch prediction system
def batch_predict_inventory(predictor, prediction_requests):
    """
    Make predictions for multiple products/dates at once
    
    Parameters:
    - predictor: InventoryPredictor instance
    - prediction_requests: List of dictionaries with prediction inputs
    
    Returns:
    - DataFrame with all predictions
    """
    results = []
    
    for request in prediction_requests:
        try:
            result = predictor.predict_simple(**request)
            
            # Flatten the result for DataFrame
            flat_result = {
                'product_id': result['input_summary']['product_id'],
                'date': result['input_summary']['date'],
                'predicted_sales': result['predicted_sales'],
                'base_demand': result['base_demand'],
                'uplift_units': result['uplift_units'],
                'uplift_percentage': result['uplift_percentage'],
                'weather_event': result['input_summary']['weather_event'],
                'festival_event': result['input_summary']['festival_event'],
                'natural_disaster': result['input_summary']['natural_disaster'],
                'economic_event': result['input_summary']['economic_event'],
                'external_intensity': result['feature_summary']['external_intensity']
            }
            results.append(flat_result)
            
        except Exception as e:
            print(f"Error predicting for {request}: {e}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Run demo
    demo_simple_predictions()