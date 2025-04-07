from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta
from datetime import datetime
import calendar
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO
import base64

app = Flask(__name__)

# Load the saved model and supporting files
try:
    with open('crime_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('crime_mapping.pkl', 'rb') as f:
        crime_mapping = pickle.load(f)
    
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    print("Model and supporting files loaded successfully")
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    crime_mapping = {}
    scaler = None

# Get the feature names from your model
if model is not None:
    all_features = model.feature_names_in_
else:
    # If model loading failed, define a placeholder list
    all_features = ['Time_Hour', 'Area ID', 'Reporting District', 'Victim Age', 
                   'Reported_Year', 'Reported_Month', 'Reported_Day', 'Reported_DayOfWeek',
                   'Occurred_Year', 'Occurred_Month', 'Occurred_Day', 'Occurred_DayOfWeek']
    all_features.extend([f'col_{i}' for i in range(20)])  # Add some placeholders

# Function to generate predictions for future timeframe
def predict_for_timeframe(start_date, end_date, area_ids=None):
    """
    Generate crime predictions for a specific timeframe and set of areas
    
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
                features = pd.DataFrame({feature: [0] for feature in all_features})
                
                # Fill relevant features
                if 'Time_Hour' in all_features:
                    features['Time_Hour'] = hour
                if 'Area ID' in all_features:
                    features['Area ID'] = area_id
                if 'Reported_Year' in all_features:
                    features['Reported_Year'] = date.year
                if 'Reported_Month' in all_features:
                    features['Reported_Month'] = date.month
                if 'Reported_Day' in all_features:
                    features['Reported_Day'] = date.day
                if 'Reported_DayOfWeek' in all_features:
                    features['Reported_DayOfWeek'] = date.dayofweek
                if 'Occurred_Year' in all_features:
                    features['Occurred_Year'] = date.year
                if 'Occurred_Month' in all_features:
                    features['Occurred_Month'] = date.month
                if 'Occurred_Day' in all_features:
                    features['Occurred_Day'] = date.day
                if 'Occurred_DayOfWeek' in all_features:
                    features['Occurred_DayOfWeek'] = date.dayofweek
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Get predictions and probabilities
                crime_code = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                max_prob = np.max(probabilities)
                
                # Find top 3 crime types
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_crimes = [model.classes_[idx] for idx in top_indices]
                top_probs = [probabilities[idx] for idx in top_indices]
                
                # Create prediction record
                predictions.append({
                    'date': date,
                    'area_id': area_id,
                    'hour': hour,
                    'predicted_crime_code': crime_code,
                    'predicted_crime_description': crime_mapping.get(crime_code, "Unknown"),
                    'confidence': max_prob,
                    'top_crime_1': top_crimes[0],
                    'top_crime_1_desc': crime_mapping.get(top_crimes[0], "Unknown"),
                    'top_crime_1_prob': top_probs[0],
                    'top_crime_2': top_crimes[1],
                    'top_crime_2_desc': crime_mapping.get(top_crimes[1], "Unknown"),
                    'top_crime_2_prob': top_probs[1],
                    'top_crime_3': top_crimes[2],  
                    'top_crime_3_desc': crime_mapping.get(top_crimes[2], "Unknown"),
                    'top_crime_3_prob': top_probs[2]
                })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions)
    return predictions_df

# Function to generate monthly prediction report
def generate_monthly_predictions(year, month, area_ids=None):
    """Generate crime predictions for a specific month and year"""
    # Get the start and end dates for the month
    start_date = datetime(year, month, 1)
    end_date = start_date + relativedelta(months=1) - relativedelta(days=1)
    
    # Generate predictions
    predictions = predict_for_timeframe(start_date, end_date, area_ids)
    return predictions

# Function to create plots for the report
def create_report_plots(predictions_df):
    """Create visualizations for the monthly report"""
    plots = {}
    
    # Create a BytesIO object for each plot
    # 1. Crime distribution by area
    plt.figure(figsize=(10, 6))
    area_counts = predictions_df.groupby('area_id')['predicted_crime_code'].count()
    sns.barplot(x=area_counts.index, y=area_counts.values)
    plt.title('Predicted Crime Distribution by Area')
    plt.xlabel('Area ID')
    plt.ylabel('Number of Predicted Crimes')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['area_distribution'] = buffer
    plt.close()
    
    # 2. Top crime types for the month
    plt.figure(figsize=(10, 6))
    crime_counts = predictions_df['predicted_crime_description'].value_counts().head(10)
    sns.barplot(x=crime_counts.values, y=crime_counts.index)
    plt.title('Top 10 Predicted Crime Types')
    plt.xlabel('Number of Predictions')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['crime_types'] = buffer
    plt.close()
    
    # 3. Crime distribution by time of day
    plt.figure(figsize=(10, 6))
    hour_counts = predictions_df.groupby('hour')['predicted_crime_code'].count()
    sns.barplot(x=hour_counts.index, y=hour_counts.values)
    plt.title('Predicted Crime Distribution by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Predicted Crimes')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['hour_distribution'] = buffer
    plt.close()
    
    # 4. Heatmap of crime by day and area
    if len(predictions_df) > 0:  # Check if we have enough data
        pivot_data = predictions_df.pivot_table(
            index=predictions_df['date'].dt.day,
            columns='area_id',
            values='predicted_crime_code',
            aggfunc='count',
            fill_value=0
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, cmap='YlOrRd')
        plt.title('Predicted Crime Heatmap by Day and Area')
        plt.xlabel('Area ID')
        plt.ylabel('Day of Month')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plots['day_area_heatmap'] = buffer
        plt.close()
    
    return plots

def generate_pdf_report(predictions_df, year, month, plots):
    """Generate a PDF report based on the monthly predictions"""
    # Create a file to save the report
    month_name = calendar.month_name[month]
    report_filename = f"crime_prediction_report_{year}_{month}.pdf"
    doc = SimpleDocTemplate(report_filename, pagesize=letter)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    normal_style = styles['Normal']
    
    # Create custom styles
    subtitle_style = ParagraphStyle(
        'subtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12
    )
    
    # Build report content
    content = []
    
    # Title
    content.append(Paragraph(f"Crime Prediction Report", title_style))
    content.append(Paragraph(f"{month_name} {year}", subtitle_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Introduction
    content.append(Paragraph("Executive Summary", heading_style))
    total_crimes = len(predictions_df)
    highest_risk_area = predictions_df.groupby('area_id')['predicted_crime_code'].count().idxmax()
    most_common_crime = predictions_df['predicted_crime_description'].value_counts().index[0]
    highest_risk_time = predictions_df.groupby('hour')['predicted_crime_code'].count().idxmax()
    
    summary_text = f"""
    This report presents crime predictions for {month_name} {year}. The AI model predicts 
    a total of {total_crimes} potential criminal incidents across all areas analyzed. 
    Area {highest_risk_area} shows the highest risk level. The most commonly predicted 
    crime type is {most_common_crime}, and the highest risk time of day is around {highest_risk_time}:00 hours.
    """
    content.append(Paragraph(summary_text, normal_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Area Distribution Chart
    if 'area_distribution' in plots:
        content.append(Paragraph("Crime Distribution by Area", heading_style))
        area_img = Image(plots['area_distribution'], width=6*inch, height=3*inch)
        content.append(area_img)
        content.append(Spacer(1, 0.25*inch))
    
    # Crime Types Chart
    if 'crime_types' in plots:
        content.append(Paragraph("Top Predicted Crime Types", heading_style))
        types_img = Image(plots['crime_types'], width=6*inch, height=3*inch)
        content.append(types_img)
        content.append(Spacer(1, 0.25*inch))
    
    # Hour Distribution Chart
    if 'hour_distribution' in plots:
        content.append(Paragraph("Crime Distribution by Time of Day", heading_style))
        hour_img = Image(plots['hour_distribution'], width=6*inch, height=3*inch)
        content.append(hour_img)
        content.append(Spacer(1, 0.25*inch))
    
    # Day-Area Heatmap
    if 'day_area_heatmap' in plots:
        content.append(Paragraph("Daily Crime Risk by Area", heading_style))
        heatmap_img = Image(plots['day_area_heatmap'], width=6*inch, height=3*inch)
        content.append(heatmap_img)
        content.append(Spacer(1, 0.25*inch))
    
    # High risk areas and times
    content.append(Paragraph("High Risk Predictions", heading_style))
    
    # Get top 5 highest confidence predictions
    high_risk = predictions_df.sort_values('confidence', ascending=False).head(10)
    
    # Create a table for high risk predictions
    table_data = [['Date', 'Area', 'Time', 'Predicted Crime', 'Confidence']]
    for _, row in high_risk.iterrows():
        table_data.append([
            row['date'].strftime('%Y-%m-%d'),
            f"Area {row['area_id']}",
            f"{row['hour']}:00",
            row['predicted_crime_description'],
            f"{row['confidence']:.2f}"
        ])
    
    high_risk_table = Table(table_data, colWidths=[1*inch, 0.8*inch, 0.8*inch, 2.5*inch, 0.9*inch])
    high_risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    content.append(high_risk_table)
    content.append(Spacer(1, 0.25*inch))
    
    # Recommendations
    content.append(Paragraph("Recommendations", heading_style))
    
    recommendations = f"""
    Based on the predictions, we recommend:
    
    1. Increase patrols in Area {highest_risk_area}, especially around {highest_risk_time}:00 hours.
    
    2. Allocate resources to address {most_common_crime}, which is predicted to be the most common crime type.
    
    3. Consider seasonal factors affecting crime patterns in {month_name}.
    
    4. Coordinate with community resources to implement preventative measures in high-risk areas.
    """
    content.append(Paragraph(recommendations, normal_style))
    
    # Build the PDF
    doc.build(content)
    return report_filename

# Routes
@app.route('/')
def index():
    # Get current month and year
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    # Calculate next month for prediction
    if current_month == 12:
        next_month = 1
        next_year = current_year + 1
    else:
        next_month = current_month + 1
        next_year = current_year
        
    # Create list of months and years for the dropdown
    months = [(i, calendar.month_name[i]) for i in range(1, 13)]
    years = list(range(current_year, current_year + 3))
    
    return render_template('index.html', 
                          current_month=current_month,
                          current_year=current_year,
                          next_month=next_month,
                          next_year=next_year,
                          months=months, 
                          years=years)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    # Get parameters from form
    year = int(request.form.get('year'))
    month = int(request.form.get('month'))
    area_ids = request.form.getlist('area_ids')
    
    # Convert area_ids to integers if provided
    if area_ids:
        area_ids = [int(area_id) for area_id in area_ids]
    else:
        area_ids = None
    
    try:
        # Generate predictions
        predictions_df = generate_monthly_predictions(year, month, area_ids)
        
        # Create plots
        plots = create_report_plots(predictions_df)
        
        # Generate PDF report
        report_filename = generate_pdf_report(predictions_df, year, month, plots)
        
        # Return the filename for download
        return jsonify({
            'success': True, 
            'filename': report_filename,
            'download_url': url_for('download_report', filename=report_filename)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_report/<filename>')
def download_report(filename):
    # Return the file for download
    return send_file(filename, as_attachment=True)

@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    # Get parameters from query string
    try:
        year = int(request.args.get('year', datetime.now().year))
        month = int(request.args.get('month', datetime.now().month))
        
        # Generate predictions
        predictions_df = generate_monthly_predictions(year, month)
        
        # Return the predictions as JSON
        return jsonify({
            'success': True,
            'year': year,
            'month': month,
            'predictions': predictions_df.to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create directories for reports if they don't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    app.run(debug=True)