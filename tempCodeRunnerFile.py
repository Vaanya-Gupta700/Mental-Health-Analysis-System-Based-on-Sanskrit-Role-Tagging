from Preprocessing import clean_text, split_into_sentences
from DomainDetection import detect_domain
from Mapping2 import analyze_structural_sentiment
import pandas as pd
import math

def run_main():
    all_plot_data = []
    df = pd.read_csv("mental_health_data.csv") 
