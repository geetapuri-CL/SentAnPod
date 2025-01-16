import pandas as pd

# File path to the input file
file_path = "/data/Sentences_75Agree.txt"

# Read the file line by line and process it
sentences = []
sentiments = []

with open(file_path, "r", encoding="iso-8859-1") as f:
    for line in f:
        # Split the line into sentence and sentiment
        if "@" in line:  # Check for valid structure
            parts = line.rsplit("@", 1)  # Split on the last occurrence of "@"
            sentences.append(parts[0].strip())  # The sentence
            sentiments.append(parts[1].strip())  # The sentiment

# Create a DataFrame
df = pd.DataFrame({"sentence": sentences, "sentiment": sentiments})

# Map sentiments to emotions
sentiment_to_emotion = {
    "positive": "joy",
    "neutral": "neutral",
    "negative": "fear"  # You can also use "sadness" based on your interpretation
}

df["emotion"] = df["sentiment"].map(sentiment_to_emotion)

# Save the processed DataFrame to a CSV file
output_path = "/data/FinancialPhraseBank.csv"
df.to_csv(output_path, index=False)

print(f"Processed file saved to {output_path}")
print(df.head())  # Preview the first few rows
