# Mental-Health-Analysis-System-Based-on-Sanskrit-Role-Tagging
A Streamlit-based analytics project for monitoring longitudinal mental health trends from text data. It combines Dependeny Parsing, Time-series Analysis, and Sanskrit-inspired Structural Linguistics (Karaka Theory) to generate patient-level insights from journal, therapy or clinical entries.

## Features
- Domain Detection (Diary/ Therapy/Clinical) using Transformers
- Karaka-based Structural Sentiment Analysis
- Longitudinal Tracking of patient mental state
- Patient-specific reports and charts
- 7-Day Risk Prediction using Weighted Regression
- Interactive dashboard using Streamlit

## Technology Stack
- Python
- Streamlit
- spaCy
- Hugging Face Transformers
- scikit-learn
- Pandas, NumPy, Seaborn, Matplotlib

## Installation
1. Clone the repository.
2. Install dependencies:  
   python -m pip install -r requirements.txt
3. Install spaCy model:  
   python -m spacy download en_core_web_md
4. Run the app:  
   python -m streamlit run app.py
5. After running the command, open the local URL shown in the terminal.

## Input Dataset Format
Required CSV columns:  
Patient_ID, Date, Text_Column

Example:  
P_101,2026-04-01,I feel overwhelmed today.  
P_101,2026-04-02,I am taking control again.
