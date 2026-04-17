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
            
            # KPI Display
            display_data = predictions[predictions['Patient'] == selected_patient]
            st.metric(label="Current Agency Score", 
                      value=f"{display_data['Current_Score'].values[0]:.2f}", 
                      delta=f"{display_data['Trend_Velocity'].values[0]:.4f}")
            st.write(f"**Clinical Status:** {display_data['Clinical_Status'].values[0]}")

            st.divider()
            st.subheader(f"7-Day Predictive Agency Trend: {selected_patient}")
            
            # --- PREDICTIVE GRAPH LOGIC (PLOTLY) ---
            import plotly.graph_objects as go
            import numpy as np
            from datetime import timedelta

            # 1. Prepare Historical Data
            plot_data = df_plot[df_plot['Patient_ID'] == selected_patient].sort_values('Date')
            
            # 2. Extract Prediction Values from our Predictor
            # (Using the values already calculated in your calculate_weighted_prediction function)
            current_score = display_data['Current_Score'].values[0]
            forecast_score = display_data['7D_Forecast'].values[0]
            last_date = plot_data['Date'].max()
            future_date = last_date + timedelta(days=7)

            # 3. Build the Plotly Figure
            fig = go.Figure()

            # TRACE 1: Historical Data (Solid Teal Line)
            fig.add_trace(go.Scatter(
                x=plot_data['Date'], 
                y=plot_data['Dominance_Score'],
                mode='lines+markers',
                name='Historical Agency (Pratyakṣa)',
                line=dict(color='teal', width=3)
            ))

            # TRACE 2: Predictive Forecast (Dashed Orange Line)
            # We connect the last historical point to the future point
            fig.add_trace(go.Scatter(
                x=[last_date, future_date],
                y=[current_score, forecast_score],
                mode='lines',
                name='7D Forecast (Anumāna)',
                line=dict(color='orange', width=3, dash='dash')
            ))

            # Add Threshold Backgrounds (Tamas vs Sattva)
            fig.add_hrect(y0=0, y1=0.4, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Tamasic Zone")
            fig.add_hrect(y0=0.6, y1=1.0, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Sattvic Zone")

            fig.update_layout(
                yaxis_range=[0, 1],
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
            # ---------------------------------------

        with tab2:
            st.subheader(f"Detailed Structural Breakdown: {selected_patient}")
            target_data = predictions[predictions['Patient'] == selected_patient]
            st.dataframe(target_data, use_container_width=True)
        
            # Using columns to create a clean "Glossary" look
            g1, g2 = st.columns(2)

            with g1:
              st.write("**📊 Dominance Score (Agency Index)**")
              st.caption("""
        The final normalized value (0 to 1). 
        - **< 0.4 (Tamasic):** High passivity; patient is the *Karma* (Object) of external forces.
        - **> 0.6 (Sattvic):** High agency; patient is the *Svatantra Karta* (Independent Doer).
        """)

              st.write("**🚀 Trend Velocity**")
              st.caption("""
        Calculates the rate of change in agency over time. A positive velocity indicates a 
        linguistic shift toward self-empowerment and recovery.
        """)

            with g2:
             st.write("**🏷️ Tagging Multipliers**")
             st.caption("""
        Derived from Pāṇinian Karaka roles. 
        - **Karta (Subject):** Multiplier 1.5x (Boosts agency).
        - **Karma (Object):** Multiplier 0.8x (Dampens agency if the self is the object).
        """)

             st.write("**🧠 Risk Probability**")
        st.caption("""
        The statistical likelihood that the current linguistic structure reflects a 
        'Helplessness' state. Lower percentages correlate with higher recovery stability.
        """)

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
