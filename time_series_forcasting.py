import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import calendar
import os
import warnings
warnings.filterwarnings("ignore")

class CrimeForecastingEngine:
    """
    A class to handle time series forecasting for crime prediction
    that extends the existing RandomForest model with temporal analysis
    """
    
    def __init__(self, model_path='crime_prediction_model.pkl',
               crime_mapping_path='crime_mapping.pkl',
              scaler_path='feature_scaler.pkl'):
        """Initialize the forecasting engine with the pre-trained model"""
        # Load the classification model
        try:
            with open(model_path, 'rb') as f:
                self.clf_model = pickle.load(f)
                
            # Load crime mapping
            with open(crime_mapping_path, 'rb') as f:
                self.crime_mapping = pickle.load(f)
                
            # Load feature scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            print("Models loaded successfully")
            self.feature_names = self.clf_model.feature_names_in_
        except Exception as e:
            print(f"Error loading models: {e}")
            self.clf_model = None
            self.crime_mapping = {}
            self.scaler = None
            self.feature_names = []
            
        # Initialize time series models dictionary
        self.ts_models = {}
        
    def preprocess_historical_data(self, historical_data):
        """
        Process historical crime data to prepare it for time series analysis
        
        Parameters:
        - historical_data: DataFrame with historical crime data
        
        Returns:
        - DataFrame with aggregated time series data
        """
        # Make sure date column is datetime
        if 'date' in historical_data.columns:
            historical_data['date'] = pd.to_datetime(historical_data['date'])
        elif 'Date Occurred' in historical_data.columns:
            historical_data['date'] = pd.to_datetime(historical_data['Date Occurred'])
        
        # Create time series by grouping by date
        daily_counts = historical_data.groupby(historical_data['date'].dt.date).size()
        daily_series = pd.Series(daily_counts, index=pd.to_datetime(daily_counts.index))
        
        # Resample to ensure continuous dates
        daily_series = daily_series.resample('D').asfreq().fillna(0)
        
        return daily_series
    
    def train_time_series_model(self, area_id, crime_type, historical_data):
        """
        Train time series forecasting model for a specific area and crime type
        
        Parameters:
        - area_id: The area ID to forecast for
        - crime_type: The crime type to forecast
        - historical_data: DataFrame with historical crime data
        
        Returns:
        - Trained time series model
        """
        # Filter data for this area and crime type
        filtered_data = historical_data
        if 'Area ID' in historical_data.columns:
            filtered_data = filtered_data[filtered_data['Area ID'] == area_id]
        if 'Crime Code' in historical_data.columns:
            filtered_data = filtered_data[filtered_data['Crime Code'] == crime_type]
            
        # Process the filtered data into a time series
        crime_series = self.preprocess_historical_data(filtered_data)
        
        # Check if we have enough data
        if len(crime_series) < 30:
            print(f"Warning: Not enough historical data for area {area_id}, crime type {crime_type}")
            return None
            
        try:
            # Perform seasonal decomposition to understand patterns
            decomposition = seasonal_decompose(crime_series, model='additive', period=7)
            
            # Train ARIMA model
            # Auto-select parameters p, d, q based on data characteristics
            p, d, q = 1, 1, 1  # Default parameters
            
            # Fit the ARIMA model
            model = ARIMA(crime_series, order=(p, d, q))
            model_fit = model.fit()
            
            # Store the trained model
            model_key = f"area_{area_id}_crime_{crime_type}"
            self.ts_models[model_key] = model_fit
            
            return model_fit
            
        except Exception as e:
            print(f"Error training time series model: {e}")
            return None
            
    def forecast_crime_counts(self, area_id, crime_type, days=30):
        """
        Generate forecasts for specific area and crime type
        
        Parameters:
        - area_id: Area ID to forecast for
        - crime_type: Crime type to forecast
        - days: Number of days to forecast
        
        Returns:
        - DataFrame with forecasted counts
        """
        model_key = f"area_{area_id}_crime_{crime_type}"
        
        if model_key not in self.ts_models:
            print(f"No time series model found for {model_key}")
            return None
            
        # Get the model
        model = self.ts_models[model_key]
        
        # Generate forecast
        forecast = model.forecast(steps=days)
        forecast_df = pd.DataFrame({
            'date': pd.date_range(start=datetime.now(), periods=days),
            'area_id': area_id,
            'crime_type': crime_type,
            'crime_description': self.crime_mapping.get(crime_type, "Unknown"),
            'predicted_count': np.maximum(0, np.round(forecast))  # Ensure counts are non-negative
        })
        
        return forecast_df
        
    def forecast_all_areas(self, crime_types=None, days=30):
        """
        Generate forecasts for all trained area and crime combinations
        
        Parameters:
        - crime_types: List of crime types to forecast (None = all available)
        - days: Number of days to forecast
        
        Returns:
        - DataFrame with all forecasts
        """
        all_forecasts = []
        
        for model_key in self.ts_models:
            # Parse area and crime from key
            parts = model_key.split('_')
            area_id = int(parts[1])
            crime_type = int(parts[3])
            
            # Check if this crime type is requested
            if crime_types is not None and crime_type not in crime_types:
                continue
                
            # Get forecast for this combination
            forecast = self.forecast_crime_counts(area_id, crime_type, days)
            if forecast is not None:
                all_forecasts.append(forecast)
                
        if all_forecasts:
            return pd.concat(all_forecasts, ignore_index=True)
        else:
            return pd.DataFrame()
            
    def combine_forecasts_with_classification(self, start_date, end_date, area_ids=None):
        """
        Combine time series forecasts with classification model predictions
        
        Parameters:
        - start_date: datetime object for the start of the prediction period
        - end_date: datetime object for the end of the prediction period
        - area_ids: list of area IDs to include (None = all areas)
        
        Returns:
        - DataFrame with enhanced predictions
        """
        # First get the classifier predictions
        if self.clf_model is None:
            print("Classification model not available")
            return None
            
        # Get classification predictions using predict_for_timeframe from original code
        classification_predictions = self.predict_for_timeframe(start_date, end_date, area_ids)
        
        # Get time series predictions for the same period
        days = (end_date - start_date).days + 1
        ts_predictions = self.forecast_all_areas(days=days)
        
        if ts_predictions.empty:
            print("No time series forecasts available, returning classification only")
            return classification_predictions
            
        # Merge predictions to enhance confidence scores
        # Group time series predictions by date and area
        ts_daily_totals = ts_predictions.groupby(['date', 'area_id'])['predicted_count'].sum().reset_index()
        
        # Convert date column in classification predictions if needed
        if 'date' in classification_predictions.columns:
            classification_predictions['date'] = pd.to_datetime(classification_predictions['date'])
        
        # Merge the datasets
        merged_predictions = classification_predictions.merge(
            ts_daily_totals, 
            on=['date', 'area_id'], 
            how='left'
        )
        
        # Fill NaN values
        merged_predictions['predicted_count'] = merged_predictions['predicted_count'].fillna(0)
        
        # Adjust confidence based on time series prediction
        # Higher predicted counts from time series model should boost confidence
        # This is a simple linear adjustment - you can customize this
        merged_predictions['adjusted_confidence'] = merged_predictions['confidence'] * (1 + merged_predictions['predicted_count'] / 100)
        
        # Cap adjusted confidence at 1.0
        merged_predictions['adjusted_confidence'] = merged_predictions['adjusted_confidence'].clip(0, 1)
        
        return merged_predictions
    
    def predict_for_timeframe(self, start_date, end_date, area_ids=None):
        """
        Generate crime predictions for a specific timeframe and set of areas
        Uses the classifier model for predictions
        
        Parameters:
        - start_date: datetime object for the start of the prediction period
        - end_date: datetime object for the end of the prediction period
        - area_ids: list of area IDs to include (None = all areas)
        
        Returns:
        - DataFrame with predictions
        """
        # Create date range for the prediction period
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Define areas to predict for
        if area_ids is None:
            # Use most common areas from training if available, otherwise use 1-20
            area_ids = list(range(1, 21))
        
        # Create combinations of dates and areas
        predictions = []
        
        for date in date_range:
            for area_id in area_ids:
                # For each hour of the day (or use specific hours with higher crime rates)
                for hour in [0, 6, 12, 18, 21, 23]:  # Morning, midday, evening, night
                    # Create feature vector for this scenario
                    features = pd.DataFrame({feature: [0] for feature in self.feature_names})
                    
                    # Fill relevant features
                    if 'Time_Hour' in self.feature_names:
                        features['Time_Hour'] = hour
                    if 'Area ID' in self.feature_names:
                        features['Area ID'] = area_id
                    if 'Reported_Year' in self.feature_names:
                        features['Reported_Year'] = date.year
                    if 'Reported_Month' in self.feature_names:
                        features['Reported_Month'] = date.month
                    if 'Reported_Day' in self.feature_names:
                        features['Reported_Day'] = date.day
                    if 'Reported_DayOfWeek' in self.feature_names:
                        features['Reported_DayOfWeek'] = date.dayofweek
                    if 'Occurred_Year' in self.feature_names:
                        features['Occurred_Year'] = date.year
                    if 'Occurred_Month' in self.feature_names:
                        features['Occurred_Month'] = date.month
                    if 'Occurred_Day' in self.feature_names:
                        features['Occurred_Day'] = date.day
                    if 'Occurred_DayOfWeek' in self.feature_names:
                        features['Occurred_DayOfWeek'] = date.dayofweek
                    
                    # Scale features
                    features_scaled = self.scaler.transform(features)
                    
                    # Get predictions and probabilities
                    crime_code = self.clf_model.predict(features_scaled)[0]
                    probabilities = self.clf_model.predict_proba(features_scaled)[0]
                    max_prob = np.max(probabilities)
                    
                    # Find top 3 crime types
                    top_indices = np.argsort(probabilities)[-3:][::-1]
                    top_crimes = [self.clf_model.classes_[idx] for idx in top_indices]
                    top_probs = [probabilities[idx] for idx in top_indices]
                    
                    # Create prediction record
                    predictions.append({
                        'date': date,
                        'area_id': area_id,
                        'hour': hour,
                        'predicted_crime_code': crime_code,
                        'predicted_crime_description': self.crime_mapping.get(crime_code, "Unknown"),
                        'confidence': max_prob,
                        'top_crime_1': top_crimes[0],
                        'top_crime_1_desc': self.crime_mapping.get(top_crimes[0], "Unknown"),
                        'top_crime_1_prob': top_probs[0],
                        'top_crime_2': top_crimes[1],
                        'top_crime_2_desc': self.crime_mapping.get(top_crimes[1], "Unknown"),
                        'top_crime_2_prob': top_probs[1],
                        'top_crime_3': top_crimes[2],  
                        'top_crime_3_desc': self.crime_mapping.get(top_crimes[2], "Unknown"),
                        'top_crime_3_prob': top_probs[2]
                    })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        return predictions_df
    
    def plot_forecast_trends(self, forecast_df, crime_types=None, areas=None):
        """
        Create visualization of crime forecasts
        
        Parameters:
        - forecast_df: DataFrame with forecast data
        - crime_types: List of crime types to include (None = all)
        - areas: List of areas to include (None = all)
        
        Returns:
        - Matplotlib figure
        """
        # Filter data if needed
        plot_data = forecast_df.copy()
        if crime_types is not None:
            plot_data = plot_data[plot_data['predicted_crime_code'].isin(crime_types)]
        if areas is not None:
            plot_data = plot_data[plot_data['area_id'].isin(areas)]
            
        # Group by date
        daily_counts = plot_data.groupby('date').size()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        daily_counts.plot(ax=ax)
        plt.title('Forecasted Crime Trends')
        plt.xlabel('Date')
        plt.ylabel('Predicted Incidents')
        plt.grid(True)
        
        return fig
        
    def save_models(self, output_dir='models'):
        """Save all trained time series models"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save time series models
        for model_key, model in self.ts_models.items():
            model_path = os.path.join(output_dir, f"{model_key}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
        print(f"Models saved to {output_dir}")
        
    def load_models(self, input_dir='models'):
        """Load time series models from directory"""
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} does not exist")
            return
            
        # Load all model files
        for filename in os.listdir(input_dir):
            if filename.endswith('.pkl') and filename.startswith('area_'):
                model_key = filename[:-4]  # Remove .pkl extension
                model_path = os.path.join(input_dir, filename)
                
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        self.ts_models[model_key] = model
                except Exception as e:
                    print(f"Error loading model {filename}: {e}")
                    
        print(f"Loaded {len(self.ts_models)} time series models")