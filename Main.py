from Preprocessing import clean_text, split_into_sentences
from DomainDetection import detect_domain
from Mapping2 import analyze_structural_sentiment
import pandas as pd
import math

def run_main():
    all_plot_data = []
    df = pd.read_csv("mental_health_data.csv") 

    # --- LEVEL 1: PATIENT LOOP ---
    for patient_id, patient_data in df.groupby('Patient_ID'):
        print(f"\n{'._.'*40}")
        print(f"ANALYZING PATIENT: {patient_id}")
        
        # Determine domain once per patient using their entire text
        patient_corpus = " ".join(patient_data['Text_Column'].astype(str))
        patient_global_domain = detect_domain(patient_corpus)
        
        print(f"ESTABLISHED CONTEXT: {patient_global_domain}")
        print(f"{'._.'*40}\n")
    
        # --- LEVEL 2: DATE LOOP (Must be indented!) ---
        for date, daily_group in patient_data.groupby('Date'):
            print(f"\n{'='*20} DATE: {date} | PATIENT: {patient_id} {'='*20}")
            daily_raw_score = 0
            
            # --- LEVEL 3: ENTRY/ROW LOOP ---
            for _, row in daily_group.iterrows():
                raw_entry = row['Text_Column']
                sentences = split_into_sentences(raw_entry)

                # Processes each sentence separately
                for i, sentence in enumerate(sentences):
                    clean_sent = clean_text(sentence)
                    
                    # Use the 'patient_global_domain' we found at Level 1
                    analysis_results, ds_sentence_raw = analyze_structural_sentiment(clean_sent, patient_global_domain)
                    daily_raw_score += ds_sentence_raw
        
                    # 4. Output
                    print(f"SENTENCE {i+1}: {clean_sent}")
                    print("=" * 90)
        
                    # Headers for clarity
                    print(f"{'Word':<12} | {'Role':<12} | {'Polarity':<10} | {'Mult':<8} | {'Final':<10} | {'Interlink'}")
                    print("=" * 90)

                    for item in analysis_results:
                    # Use the EXACT keys you defined in results.append
                     word = item["word"]
                     role = item["role"]
                     pol  = item["polarity"]
                     mult = item["multiplier"]
                     fin  = item["final"]
                     interlink = item["interlink"] # The dependency link
              
                 # Print row with formatted spacing
                     print(f"{word:<12} | {role:<12} | {pol:<10.2f} | {mult:<8.1f} | {fin:<10.3f} | {interlink}")
        
                     print("_" * 100)

            # dominance score(btw 0 and 1)
            paragraph_dominance = 1 / (1 + math.exp(-daily_raw_score))

            all_plot_data.append({
    "Patient_ID": patient_id,
    "Date": date,
    "Dominance_Score": paragraph_dominance,
    "Detected_Domain": patient_global_domain
})
    
            print("\n" + "="*100)
            print(f"FINAL PARAGRAPH ANALYSIS")
            print(f"Total Raw Score: {daily_raw_score:.4f}")
            print(f"OVERALL DOMINANCE SCORE (Sigmoid): {paragraph_dominance:.4f}")
            print("="*100)

    plot_df = pd.DataFrame(all_plot_data)
    plot_df.to_csv("dominance_trends.csv", index=False)
    print("\n[System] Plot data saved to dominance_trends.csv")           

if __name__ == "__main__":
    run_main()
   
