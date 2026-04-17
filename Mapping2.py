import spacy
from textblob import TextBlob
from DomainDetection import detect_domain

# Load the brain
nlp = spacy.load("en_core_web_md")
 
# references for base polarity in each domain
# weight_scale essentially decides how much the "vibe" or "emotional charge" of the words should weigh
domain_anchors = {
    "Private Diary Entry": {
       "POSITIVE": nlp("positive try change motivated calmer stable hopeful focus aware present clear confident manage improve accomplish control handle interact comfortably enjoy shared connected"),
        "NEGATIVE": nlp("low impossible panic defeat stuck fear circumstances sad dragged drained exhausted heavy overwhelming overthinking intense crowded unmanageable stuck giving-up avoiding exhausting isolated silent"),
        "weight_scale": 1.5 
    },
    "Peer Support/Therapy": {
        "POSITIVE": nlp("shared supporting helping friend each other understood growning connected listened helpful"),
        "NEGATIVE": nlp("isolated drown alone struggle pain lost ignored silent misunderstood"),
        "weight_scale": 1.0
    },
    "Clinical Mental Health Analysis": {
        "POSITIVE": nlp("stable improved strong functional consistent walk recovering better "),
        "NEGATIVE": nlp("relapse darkness drown overwhelming controlling symptomatic negative episodic chronic suffocation insomnia unable "),
        "weight_scale": 0.5 
    }
}

def analyze_structural_sentiment(sentence_text, top_domain=None):
    # If no domain was passed, use your Hugging Face script to find it
    if top_domain is None:
        top_domain = detect_domain(sentence_text)
    
    print(f"Analyzing using weights for: {top_domain}")
 # tokenization(parsing)
    doc = nlp(sentence_text)
    results = []
    
    # Storage for the links
    links_map = {token.text: [] for token in doc}

    # 1. First Pass: Identify the Karta-Karma bridge and Negation links
    for token in doc:
        
        # A. BASIC LINKS: Every word connects to its grammatical head (Parent)
        if token.head != token:
            t_text, h_text = token.text, token.head.text
            if h_text not in links_map[t_text]: links_map[t_text].append(h_text)
            if t_text not in links_map[h_text]: links_map[h_text].append(t_text)

        # B. THE BRIDGE: For Verbs/Aux, link Subject (Karta) to Attribute (Viseshta)
        if token.pos_ in ["VERB", "AUX"] or token.dep_ == "ROOT":
            subjects = [t for t in token.children if t.dep_ in ["nsubj", "nsubjpass"]]
            objects = [t for t in token.children if t.dep_ in ["dobj", "pobj", "attr", "acomp"]]
            
            for s in subjects:
                for o in objects:
                    # Link "I" <-> "depressed"
                    if o.text not in links_map[s.text]: links_map[s.text].append(o.text)
                    if s.text not in links_map[o.text]: links_map[o.text].append(s.text)

        # C. NEGATION PROPAGATION: Explicitly link 'not' to the Subject
        if token.dep_ == "neg":
            # Link the 'not' directly to the subjects of the word it modifies
            # e.g., in "I am not...", 'not' modifies 'am', so link 'not' to 'I'
            head_subjects = [t.text for t in token.head.children if t.dep_ in ["nsubj", "nsubjpass"]]
            for s in head_subjects:
                if token.text not in links_map[s]: links_map[s].append(token.text)
                if s not in links_map[token.text]: links_map[token.text].append(s)
            
            # Link the Verb/Aux to all participants for the full bridge
            all_participants = subjects + objects
            for participant in all_participants:
                if participant.text not in links_map[token.text]: 
                    links_map[token.text].append(participant.text)
                if token.text not in links_map[participant.text]: 
                    links_map[participant.text].append(token.text)

    # 2. Second Pass: Build the results with Role Tagging
    anchors = domain_anchors.get(top_domain)
    pos_words = [token.text.lower() for token in anchors["POSITIVE"]]
    neg_words = [token.text.lower() for token in anchors["NEGATIVE"]]

    # STEP A: Assign Roles to EVERY token first
    # Paninian Mapping
    token_roles = {}
    for token in doc:
        role = "Other"
        if token.dep_ == "nsubjpass": role = "Karma"
        elif token.dep_ == "nsubj" or (token.dep_ == "pobj" and token.head.dep_ == "agent"): role = "Karta"
        elif token.pos_ == "VERB": role = "Kriya"
        elif token.pos_ == "ADJ": role = "Viseshta"
        token_roles[token.i] = role # Use index to avoid duplicate word issues

    # STEP B: Calculate Scores
    for token in doc:
        role = token_roles[token.i]
        content_pos = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
        
        # 1. Polarity Logic (Stays the same)
        # Dictionary-meaning similarity-backup sentiment
        if token.is_stop or token.is_punct or token.pos_ not in content_pos:
            polarity = 0.0
        else:
            token_text = token.text.lower()
            if token_text in neg_words:
                polarity = -0.5 * anchors.get("weight_scale", 1.0)
            elif token_text in pos_words:
                polarity = 0.5 * anchors.get("weight_scale", 1.0)
            else:
                pos_sim = token.similarity(anchors["POSITIVE"])
                neg_sim = token.similarity(anchors["NEGATIVE"])
                if pos_sim > 0.3 or neg_sim > 0.3:
                    polarity = (pos_sim if pos_sim > neg_sim else -neg_sim)
                else:
                    polarity = TextBlob(token.text).sentiment.polarity

        # 2. Multiplier & Role Weight (Refined)
        interlink_list = links_map.get(token.text, [])
        
        # Standardize search for interlinks to handle case sensitivity
        interlink_lower = [w.lower() for w in interlink_list]
        
        is_self = any(u in interlink_lower for u in ["i", "me", "my", "mine", "myself"])
        is_external = any(o in interlink_lower for o in ["he", "she", "they", "circumstances", "brain"])

        W_MAP = {"Karta": 1.5, "Kriya": 0.5, "Karma": 0.8, "Viseshta": 0.7, "Other": 0.6}
        
        # Sva-Karta vs Para-Karta Logic
        if role == "Karta":
            # Direct check for user as the doer
            if token.text.lower() in ["i", "we"]:
                role_weight = 1.5
            else:
                role_weight = 0.4 # Penalty for external Karta
        else:
            role_weight = W_MAP.get(role, 0.6)

        # Context Multiplier: Ownership (1.5) vs Displacement (0.5)
        context_mult = 1.5 if is_self else (0.5 if is_external else 0.8)

        # FINAL CALCULATION
        final_score = polarity * context_mult * role_weight

         # Store result   
        results.append({
            "word": token.text,
            "role": role,
            "polarity": polarity,
            "multiplier": context_mult * role_weight,
            "final": final_score,
            "interlink": ", ".join(interlink_list)
        })

    # 3. Summation
    ds_raw = sum(item['final'] for item in results)
    ds_raw = 0.0
    for item in results:
        ds_raw += float(item['final'])

    return results, ds_raw
    
