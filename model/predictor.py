from config.config import FORECAST_DAYS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

def calculate_weighted_prediction(csv_file):
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    results = []

    for patient_id, data in df.groupby('Patient_ID'):
        data = data.sort_values('Date').reset_index()
        
        # --- STEP 1: DEFINE DEFAULTS AT THE START ---
        slope = 0
        forecast_7d = data['Dominance_Score'].iloc[-1] # Default to current score
        risk_probability = 0.5
        status = "Insufficient Data"
        
        days = (data['Date'] - data['Date'].min()).dt.days.values.reshape(-1, 1)
        scores = data['Dominance_Score'].values
        
        # --- STEP 2: PERFORM CALCULATIONS ---
        if len(days) > 1:
            # 1. Establish Confidence based on Data Size
            # We need at least 7-10 days for high confidence. 
            # Below 5 days, we are "skeptical" of the trend.
            confidence = min(1.0, len(days) / 7.0) 
            
            # 2. Existing Regression Math
            weights = np.exp(0.1 * np.arange(len(days)))
            model = LinearRegression()
            model.fit(days, scores, sample_weight=weights)
            
            slope = model.coef_[0]
            last_day = days[-1][0]
            raw_forecast = model.predict([[last_day + FORECAST_DAYS]])[0]
            
            # 3. THE FIX: Confidence-Weighted Smoothing
            # If confidence is low, we pull the forecast closer to the current score
            # Forecast = (Trend * Confidence) + (Current * (1 - Confidence))
            forecast_7d = (raw_forecast * confidence) + (scores[-1] * (1 - confidence))
            forecast_7d = max(0.01, min(0.99, forecast_7d))
            
            # 4. Composite Risk Logic
            risk_raw = 5 * (0.5 - forecast_7d)
            
            # If data is sparse, we suppress high risk warnings to avoid false alarms
            if confidence < 0.5:
                risk_raw -= 1.5 
                
            risk_probability = 1 / (1 + math.exp(-risk_raw))
            
            #------------------- Status based on confidence
            current_score = scores[-1]
            
            if confidence < 0.4:
                status = "Initial Assessment (Low Confidence)"
            elif current_score < 0.45:
                # Even if the slope is positive, they are still in a passive state
                if slope > 0.05:
                    status = "Emerging from Passive State"
                else:
                    status = "At Risk: Persistent Passivity"
            elif 0.45 <= current_score <= 0.65:
                if slope < -0.05:
                    status = "Warning: Fluctuating / Relapsing"
                elif slope > 0.05:
                    status = "Steady Improvement"
                else:
                    status = "Stable / Maintenance"
            else: # High Agency (Sattvic)
                if slope < -0.10:
                    status = "Warning: Sudden Agency Drop"
                else:
                    status = "Stable / High Recovery"
         #-----------------------------------           

        # --- STEP 3: APPEND (forecast_7d is now guaranteed to exist) ---
        results.append({
            "Patient": patient_id,
            "Trend_Velocity": round(slope, 4),
            "Current_Score": round(scores[-1], 4),
            "7D_Forecast": round(forecast_7d, 4),
            "Risk_Prob": f"{risk_probability*100:.1f}%",
            "Clinical_Status": status
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    pred_df = calculate_weighted_prediction("dominance_trends.csv")
    print("\n" + "*-"*40)
    print("SANSKRIT-AI CLINICAL PREDICTION REPORT (Weighted Regression)")
    print("*-"*40)
    print(pred_df.to_string(index=False))
