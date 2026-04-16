import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_separate_patient_trends(csv_file):
    # 1. Load data
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 2. Initialize the FacetGrid
    # col="Patient_ID" tells it to make a new column/graph for each patient
    # col_wrap=3 means it will put 3 graphs per row, then start a new row
    g = sns.FacetGrid(df, col="Patient_ID", col_wrap=3, height=4, aspect=1.2)

    # 3. Map the plot to the grid
    # We use 'marker="o"' so you can still see individual days
    g.map(sns.lineplot, "Date", "Dominance_Score", marker="o", color="teal")

    # 4. Add the Neutral Threshold (0.5) to EVERY separate graph
    g.map(plt.axhline, y=0.5, color="red", linestyle="--", alpha=0.5)

    # 5. Formatting
    g.set_axis_labels("Date", "Dominance Score")
    g.set_titles("Patient: {col_name}") # Titles each graph with the ID
    g.set(ylim=(0, 1)) # Keep Sigmoid scale consistent
    
    # Rotate dates so they don't overlap
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.tight_layout()
    plt.savefig("separate_patient_analysis.png")
    print("[System] Separate graphs saved as separate_patient_analysis.png")
    plt.show()

if __name__ == "__main__":
    plot_separate_patient_trends("dominance_trends.csv")