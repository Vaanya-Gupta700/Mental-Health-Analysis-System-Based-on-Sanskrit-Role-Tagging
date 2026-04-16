import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Main import run_main 
from Predictor import calculate_weighted_prediction
from DomainDetection import detect_domain

st.set_page_config(page_title="Sanskrit-AI Dashboard", layout="wide")

# We move the title inside the logic to keep the home page clean
st.sidebar.header("Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload mental_health_data.csv", type=["csv"])

if uploaded_file:
    with open("mental_health_data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # --- RUN GLOBAL ANALYSIS ---
    with st.spinner('Running Karaka Analysis...'):
        run_main() 
        predictions = calculate_weighted_prediction("dominance_trends.csv")
        df_plot = pd.read_csv("dominance_trends.csv")
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])

    # --- HOME PAGE: PATIENT SELECTION PORTAL ---
    # We define the list here to use in the main page selector
    patient_ids = df_plot['Patient_ID'].unique().tolist()
    
    # Use a session state or a simple selectbox as the "Gatekeeper"
    st.title("🏥 Patient Report Portal")
    st.markdown("Select a patient below to enter their specific structural analysis and longitudinal report.")
    
    selected_patient = st.selectbox(
        "Select Patient ID to Begin:", 
        ["-- Select Patient --"] + patient_ids,
        index=0
    )

    st.divider()

    # --- CONDITIONAL LOGIC: IF NO PATIENT SELECTED (HOME PAGE) ---
    if selected_patient == "-- Select Patient --":
        st.info("👈 Please select a Patient ID from the dropdown menu above to generate the Karaka Analysis report.")
        
        # High-level overview for the Home Page
        st.subheader("Global Clinical Summary")
        st.dataframe(predictions[['Patient', 'Clinical_Status', 'Risk_Prob']], use_container_width=True)

    # --- CONDITIONAL LOGIC: IF PATIENT SELECTED (REPORT PAGE) ---
    else:
        st.title(f"📊 Structural Report: {selected_patient}")
        
        # 1. RETRIEVE DATA FOR THIS PATIENT
        # We get the plot data and the domain name from the CSV we saved in Main.py
        plot_data = df_plot[df_plot['Patient_ID'] == selected_patient]
        
        # Pull the domain from the first row of this patient's data
        # (Assuming you saved 'Detected_Domain' in Main.py as we discussed)
        if 'Detected_Domain' in plot_data.columns:
            patient_global_domain = plot_data['Detected_Domain'].iloc[0]
        else:
            patient_global_domain = "Unknown Context"

        # 2. DISPLAY DOMAIN SECTION
        col1, col2 = st.columns([3, 1])

        with col1:
            st.info(f"**Detected Context:** {patient_global_domain}")

        with col2:
            # Color coding based on domain
            if "Clinical" in patient_global_domain:
                st.error("🔒 High Precision Mode")
            else:
                st.success("📖 Narrative Mode")

        # 3. NOW PROCEED TO TABS (Ensure these are indented correctly!)
        tab1, tab2 = st.tabs(["📈 Daily Trends", "🕉️ Structural Analysis"])

        

        with tab1:
            st.subheader("Current Patient Status")
            
            # --- NEW FILTERED KPI LOGIC ---
            # Filter the predictions dataframe to only show the selected patient
            display_data = predictions[predictions['Patient'] == selected_patient]
            
            # Display KPI for ONLY the selected patient
            cols = st.columns(1) # We only need one column now
            for i, row in display_data.iterrows():
                with cols[0]:
                    st.metric(label=f"Patient {row['Patient']}", 
                              value=f"{row['Current_Score']:.2f}", 
                              delta=row['Trend_Velocity'])
                    st.write(f"**Status:** {row['Clinical_Status']}")
            # ------------------------------

            st.divider()
            st.subheader(f"Longitudinal Dominance Trends: {selected_patient}")
            
            # Filter the graph for the selected patient
            plot_data = df_plot[df_plot['Patient_ID'] == selected_patient]

            fig = sns.FacetGrid(plot_data, col="Patient_ID", col_wrap=2, height=4, aspect=1.5)
            fig.map(sns.lineplot, "Date", "Dominance_Score", marker="o", color="teal")
            fig.map(plt.axhline, y=0.5, color="red", linestyle="--", alpha=0.5)
            for ax in fig.axes.flat:
                for label in ax.get_xticklabels(): label.set_rotation(45)
            st.pyplot(fig.figure)

        with tab2:
            st.subheader(f"Detailed Structural Breakdown: {selected_patient}")
            target_data = predictions[predictions['Patient'] == selected_patient]
            st.dataframe(target_data, use_container_width=True)
            
            st.write("### 🕉️ Karaka Mapping Insights")
            st.info(f"Showing the latest linguistic agency markers for {selected_patient}. The scores reflect the integration of Tagging Multipliers and the Pāṇinian Role Formula.")

        st.divider()
        with st.expander(f"📄 View Raw Source Data for {selected_patient}"):
            st.write(f"The following transcripts from **mental_health_data.csv** were used to generate this Karaka Agency Report:")
            
            # Load the original data to show the text
            df_raw = pd.read_csv("mental_health_data.csv")
            patient_text_data = df_raw[df_raw['Patient_ID'] == selected_patient][['Date', 'Text_Column']]
            
            # Display it as a clean table
            st.table(patient_text_data)
else:
    st.title("🕉️ Sanskrit-AI: Longitudinal Mental Health Monitor")
    st.info("Please upload the mental_health_data.csv file from the sidebar to begin analysis.")