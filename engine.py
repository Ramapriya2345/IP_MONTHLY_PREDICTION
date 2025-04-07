import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from crime_forecasting_engine import CrimeForecastingEngine

def run_forecasting():
    """Main function to demonstrate the crime forecasting engine"""
    print("Starting Crime Forecasting System")
    
    # Initialize the forecasting engine
    engine = CrimeForecastingEngine()
    
    # Check if models loaded successfully
    if engine.clf_model is None:
        print("Error: Required models not found. Make sure the model files exist in the current directory.")
        return
    
    # Load historical data
    # This needs to be customized based on your actual data file
    try:
        print("Loading historical crime data...")
        historical_data = pd.read_csv('crime-data-from-2010-to-present.csv')  # Replace with your dataset path
        print(f"Loaded {len(historical_data)} records")
    except Exception as e:
        print(f"Error loading historical data: {e}")
        print("Using example data for demonstration")
        # Create some sample data for demo purposes
        historical_data = create_sample_data()
    
    # Train time series models for common crime types in several areas
    print("\nTraining time series models...")
    
    # Get top crime types
    if 'Crime Code' in historical_data.columns:
        top_crimes = historical_data['Crime Code'].value_counts().head(5).index.tolist()
    else:
        top_crimes = [624, 510, 440, 330, 626]  # Example crime codes
    
    # Get areas
    if 'Area ID' in historical_data.columns:
        areas = historical_data['Area ID'].unique()[:5]  # First 5 areas
    else:
        areas = [1, 2, 3, 4, 5]  # Example area IDs
    
    # Train models for each area and crime type combination
    for area in areas:
        for crime in top_crimes:
            print(f"Training model for Area {area}, Crime {crime}")
            engine.train_time_series_model(area, crime, historical_data)
    
    # Generate combined forecasts for next 30 days
    print("\nGenerating forecasts...")
    start_date = datetime.now()
    end_date = start_date + timedelta(days=30)
    
    forecasts = engine.combine_forecasts_with_classification(start_date, end_date, area_ids=areas)
    
    if forecasts is not None and not forecasts.empty:
        print(f"Generated {len(forecasts)} predictions")
        
        # Display sample of predictions
        print("\nSample predictions:")
        print(forecasts[['date', 'area_id', 'predicted_crime_description', 
                         'confidence', 'adjusted_confidence']].head(10))
        
        # Plot forecast trends
        fig = engine.plot_forecast_trends(forecasts)
        plt.savefig('crime_forecast_trends.png')
        plt.show()
        
        # Save forecasts to CSV
        forecasts.to_csv('crime_forecasts.csv', index=False)
        print("Forecasts saved to crime_forecasts.csv")
        
        # Save trained models
        engine.save_models()
    else:
        print("No forecasts generated")

def create_sample_data():
    """Create sample data for demonstration purposes"""
    # Generate dates for past year
    dates = pd.date_range(end=datetime.now(), periods=365)
    
    # Create sample DataFrame
    data = []
    
    # Common crime codes
    crime_codes = [624, 510, 440, 330, 626]
    
    # Generate random crime incidents
    for _ in range(5000):
        date = dates[np.random.randint(0, len(dates))]
        area_id = np.random.randint(1, 11)
        crime_code = np.random.choice(crime_codes)
        
        data.append({
            'Date Occurred': date,
            'Area ID': area_id,
            'Crime Code': crime_code,
            'Victim Age': np.random.randint(18, 65)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    run_forecasting()