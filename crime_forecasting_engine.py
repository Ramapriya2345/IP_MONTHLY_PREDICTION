import os
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class CrimeForecastingEngine:
    def __init__(self):
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}  # (area, crime_code) -> model
        self.load_models()

        # Load classifier and scaler
        self.clf_model = self.load_pickle_file('crime_prediction_model.pkl')
        self.scaler = self.load_pickle_file('feature_scaler.pkl')
        self.crime_mapping = self.load_pickle_file('crime_mapping.pkl')

    def load_pickle_file(self, file_path):
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_models(self):
        for key, model in self.models.items():
            area, crime_code = key
            filename = f'{self.model_dir}/model_area_{area}_crime_{crime_code}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)

    def load_models(self):
        if not os.path.exists(self.model_dir):
            return
        for file in os.listdir(self.model_dir):
            if file.endswith('.pkl'):
                parts = file.split('_')
                try:
                    area = int(parts[2])
                    crime = int(parts[4].split('.')[0])
                    with open(os.path.join(self.model_dir, file), 'rb') as f:
                        model = pickle.load(f)
                        self.models[(area, crime)] = model
                except Exception as e:
                    print(f"Failed to load model {file}: {e}")

    def train_time_series_model(self, area_id, crime_code, data):
        df = data.copy()
        df['Date Occurred'] = pd.to_datetime(df['Date Occurred'])

        mask = (df['Area ID'] == area_id) & (df['Crime Code'] == crime_code)
        df_filtered = df.loc[mask]

        if df_filtered.empty:
            return None

        ts = df_filtered.groupby('Date Occurred').size().resample('D').sum().fillna(0)

        try:
            model = ExponentialSmoothing(ts, trend='add', seasonal=None).fit()
            self.models[(area_id, crime_code)] = model
            return model
        except Exception as e:
            print(f"Failed to train model for Area {area_id}, Crime {crime_code}: {e}")
            return None

    def combine_forecasts_with_classification(self, start_date, end_date, area_ids=None):
        if not self.models or self.clf_model is None or self.scaler is None:
            return None

        date_range = pd.date_range(start_date, end_date)
        results = []

        for (area_id, crime_code), model in self.models.items():
            if area_ids and area_id not in area_ids:
                continue

            try:
                forecast = model.forecast(len(date_range))
                for date, predicted_count in zip(date_range, forecast):
                    features = np.array([[area_id, date.dayofweek, date.month, predicted_count]])
                    features_scaled = self.scaler.transform(features)
                    pred = self.clf_model.predict(features_scaled)[0]
                    prob = self.clf_model.predict_proba(features_scaled)[0][pred]
                    description = self.crime_mapping.get(pred, "Unknown")

                    results.append({
                        'date': date,
                        'area_id': area_id,
                        'predicted_crime_code': pred,
                        'predicted_crime_description': description,
                        'confidence': round(prob, 2),
                        'adjusted_confidence': round(prob * predicted_count, 2),
                        'forecasted_count': round(predicted_count)
                    })
            except Exception as e:
                print(f"Forecasting failed for Area {area_id}, Crime {crime_code}: {e}")

        return pd.DataFrame(results)

    def plot_forecast_trends(self, forecast_df, selected_crimes=None, selected_areas=None):
        if forecast_df is None or forecast_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        grouped = forecast_df.groupby(['date', 'predicted_crime_description'])['forecasted_count'].sum().unstack().fillna(0)

        if selected_crimes:
            cols = [self.crime_mapping.get(c, str(c)) for c in selected_crimes]
            grouped = grouped[cols]

        grouped.plot(ax=ax)
        ax.set_title("Forecasted Crime Trends")
        ax.set_xlabel("Date")
        ax.set_ylabel("Crime Count")
        ax.legend(title="Crime Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        return fig
