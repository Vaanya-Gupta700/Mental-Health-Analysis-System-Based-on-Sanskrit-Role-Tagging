import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import timedelta

def plot_predictive_trends(csv_file):
    # 1. Load and Sort Data
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Patient_ID', 'Date'])

    # 2. Initialize FacetGrid
    g = sns.FacetGrid(df, col="Patient_ID", col_wrap=2, height=4, aspect=1.3)
    
    # Plot Historical Data (Solid line with markers)
    g.map(sns.lineplot, "Date", "Dominance_Score", marker="o", color="teal", label="Historical")

    # 3. Add Predictive Forecast to each subplot
    for ax, (patient_id, patient_data) in zip(g.axes.flat, df.groupby('Patient_ID')):
        # --- MATH: WEIGHTED REGRESSION ---
        # Convert dates to relative days for the X-axis
        days = (patient_data['Date'] - patient_data['Date'].min()).dt.days.values.reshape(-1, 1)
        scores = patient_data['Dominance_Score'].values
        
        if len(days) > 1:
            # Apply Exponential Weights (Recent entries influence the slope more)
            weights = np.exp(0.1 * np.arange(len(days)))
            model = LinearRegression()
            model.fit(days, scores, sample_weight=weights)
            
            # Predict scores for the next 7 days
            last_day = days[-1][0]
            future_days = np.array([[last_day], [last_day + 7]])
            predictions = model.predict(future_days)
            
            # Convert days back to Date objects for the plot
            last_date = patient_data['Date'].max()
            future_dates = [last_date, last_date + timedelta(days=7)]
            
            # Plot the Forecast line (Dashed Orange)
            ax.plot(future_dates, predictions, color="orange", linestyle="--", 
                    linewidth=2, label="7D Forecast")
        
        # 4. Add Clinical Threshold (0.5)
        ax.axhline(0.5, color="red", linestyle=":", alpha=0.4, label="Neutral Threshold")
        
        # Formatting individual subplots
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize='small')

    # Final Layout Tweaks
    g.set_axis_labels("Date", "Agency (Dominance) Score")
    g.set_titles("Patient: {col_name}")
    plt.tight_layout()
    plt.savefig("predictive_analysis.png")
    print("[System] Predictive graph saved as predictive_analysis.png")

if __name__ == "__main__":
    plot_predictive_trends("dominance_trends.csv")
