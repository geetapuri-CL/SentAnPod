import os
import pandas as pd
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

# Define keywords for each emotion
emotion_keywords = {
    "joy": ["surged", "growth", "positive", "opportunity", "profit", "success"],
    "fear": ["volatile", "risk", "uncertainty", "drop", "losses", "concern"],
    "anger": ["criticism", "backlash", "dissatisfied", "fraud", "failure"],
    "sadness": ["fell", "loss", "disappointed", "negative", "decline"]
}


def expand_keywords(base_keywords):
    expanded = set(base_keywords)
    for word in base_keywords:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().lower())
    return list(expanded)

# Expand keywords for "joy"
emotion_keywords["joy"] = expand_keywords(emotion_keywords["joy"])
emotion_keywords["fear"] = expand_keywords(emotion_keywords["fear"])
emotion_keywords["anger"] = expand_keywords(emotion_keywords["anger"])
emotion_keywords["sadness"] = expand_keywords(emotion_keywords["sadness"])

print("emotion keywords set = ", emotion_keywords)


# Directory containing text files
base_dir = "/data"

# Read all text files
raw_data = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                raw_data.extend(f.readlines())

# Convert to DataFrame
df_raw = pd.DataFrame(raw_data, columns=["text"])
print(f"Loaded {len(df_raw)} lines of text.")

def label_emotion(text):
    # Iterate over each emotion and its keywords
    for emotion, keywords in emotion_keywords.items():
        # Check if any keyword matches in the text
        if any(keyword in text.lower() for keyword in keywords):
            return emotion
    return "neutral"  # Default label if no keyword matches

# Apply the labeling function
df_raw["emotion"] = df_raw["text"].apply(label_emotion)

# Filter out sentences with "neutral" emotion
df_labeled = df_raw[df_raw["emotion"] != "neutral"]

# Check labeled data
print(df_labeled.head())
print(f"Labeled {len(df_labeled)} lines with emotions.")

# Check if we have at least 500 labeled sentences
if len(df_labeled) < 500:
    print("Not enough labeled data. Consider adding more raw text or expanding keywords.")
else:
    print("Sufficient data labeled.")

# Save to CSV
output_path = "/data/podcasts/financial_emotion_dataset.csv"
df_labeled.to_csv(output_path, index=False)
print(f"Labeled dataset saved to {output_path}.")

print(df_labeled['emotion'].value_counts())

