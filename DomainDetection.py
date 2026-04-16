from transformers import pipeline
# Hugging Face
# 1. Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def detect_domain(sentence_text):
    candidate_labels = ["Private Diary Entry", "Peer Support/Therapy", "Clinical Mental Health Analysis"]
    res = classifier(sentence_text, candidate_labels)
    
    # Capture both the top label and its corresponding top score
    top_label = res['labels'][0]
    top_score = res['scores'][0]
    
    # This prints to your terminal so you can check the accuracy
    print(f"--- Domain Check: {top_label} ({top_score:.2%}) ---")

    # Return them as a formatted string or a tuple
    return top_label


