import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from transformers import pipeline

# Base directory containing the subfolders
base_dir = "/data/podcasts"

# Initialize a list to store file content and metadata
data = []

# Traverse through all subdirectories and read .txt files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Append metadata and content
                data.append({
                    "folder": os.path.basename(root),  # Folder name
                    "file": file,                      # File name
                    "content": content                 # File content
                })

# Create a DataFrame with the collected data
df = pd.DataFrame(data)
print(f"Loaded {len(df)} text files.")
#print(df.head())



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing to the content
df['cleaned_content'] = df['content'].apply(preprocess_text)
#print(df.head())


# Load the pre-trained sarcasm detection model
sarcasm_detector = pipeline("text-classification", model="marijaras/sarcasm-detection")

# Apply sarcasm detection
df['sarcasm_label'] = df['cleaned_content'].apply(lambda x: sarcasm_detector(x)[0]['label'])
print(df[['folder', 'file', 'sarcasm_label']].head())