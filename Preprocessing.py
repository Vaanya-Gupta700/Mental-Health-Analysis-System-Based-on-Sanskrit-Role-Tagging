import spacy
import re

def clean_text(text):
    # 1. Remove punctuation
    # This regex looks for anything that isn't a letter, number, or space and deletes it
    text = re.sub(r'[^\w\s]', '', text)
    
    # 2. Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_into_sentences(text):
    # Step 2: Vibhaga (Partitioning)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

