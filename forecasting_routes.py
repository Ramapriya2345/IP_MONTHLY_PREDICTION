# Save this as forecasting_routes.py

from flask import Blueprint, request, jsonify, render_template, send_file
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from crime_forecasting_engine import CrimeForecastingEngine

# Initialize the forecasting module
forecasting_bp = Blueprint('forecasting', __name__)
engine = CrimeForecastingEngine()

@forecasting_bp.route('/train_models', methods=['POST'])
def train_models():
    """API endpoint to train time series models using historical data"""
    try:
        # Get parameters from request
        data_path = request.form.get('data_path', 'crime_data.csv')
        
        # Load data
        historical_data = pd.read_csv(data_path)
        
        # Get parameters
        top_n = int(request.form.get('top_crimes', 5))
        area_filter = request.form.getlist('areas')
        if area_filter:
            area_filter = [int(area) for area in area_filter]
        
        # Get crime types to train for
        if 'Crime Code' in historical_data.columns:
            top_crimes = historical_data['Crime Code'].value_counts().head(top_n).index.tolist()
        else:
            return jsonify({'success': False, 'error': 'No Crime Code column in data'})
        
        # Get areas to train for
        if 'Area ID' in historical_data.columns:
            if area_filter:
                areas = [a for a in area_filter if a in historical_data['Area ID'].unique()]
            else:
                areas = historical_data['Area ID'].unique()[:10]  # Limit to first 10 areas
        else:
            return jsonify({'success': False, 'error': 'No Area ID column in data'})
        
        # Train models
        trained_models = []
        for area in areas:
            for crime in top_crimes:
                model = engine.train_time_series_model(area, crime, historical_data)
                if model is not None:
                    trained_models.append(f"Area {area}, Crime {crime}")
        
        # Save models
        engine.save_models()
        
        return jsonify({
            'success': True,
            'models_trained': trained_models,
            'model_count': len(trained_models)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@forecasting_bp.route('/forecast', methods=['GET'])
def forecast():
    """API endpoint to generate forecasts"""
    try:
        # Get parameters
        days = int(request.args.get('days', 30))
        start_date = datetime.now()
        end_date = start_date + pd.Timedelta(days=days-1)
        
        area_ids = request.args.getlist('area_ids')
        if area_ids:
            area_ids = [int(area) for area in area_ids]
        
        # Generate forecasts
        forecasts = engine.combine_forecasts_with_classification(start_date, end_date, area_ids)
        
        if forecasts is None or forecasts.empty:
            return jsonify({'success': False, 'error': 'No forecasts generated'})
        
        # Convert date to string for JSON serialization
        forecasts['date'] = forecasts['date'].dt.strftime('%Y-%m-%d')
        
        return jsonify({
            'success': True,
            'forecast_count': len(forecasts),
            'forecasts': forecasts.to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@forecasting_bp.route('/forecast_chart', methods=['GET'])
def forecast_chart():
    """Generate and return a forecast chart"""
    try:
        # Get parameters
        days = int(request.args.get('days', 30))
        start_date = datetime.now()
        end_date = start_date + pd.Timedelta(days=days-1)
        
        area_ids = request.args.getlist('area_ids')
        if area_ids:
            area_ids = [int(area) for area in area_ids]
            
        crime_types = request.args.getlist('crime_types')
        if crime_types:
            crime_types = [int(crime) for crime in crime_types]
        
        # Generate forecasts
        forecasts = engine.combine_forecasts_with_classification(start_date, end_date, area_ids)
        
        if forecasts is None or forecasts.empty:
            return jsonify({'success': False, 'error': 'No forecasts generated'})
        
        # Create chart
        fig = engine.plot_forecast_trends(forecasts, crime_types, area_ids)
        
        # Save chart to memory
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        
        # Convert to base64 for embedding
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'chart_image': f"data:image/png;base64,{img_base64}"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Add this function to app.py to register the blueprint
def register_forecasting_routes(app):
    app.register_blueprint(forecasting_bp, url_prefix='/forecasting')