import os
import spacy
from transformers import pipeline
from spacy.pipeline import EntityRuler
import pandas as pd
import logging
from datetime import datetime

# Setup logging
log_folder = "/geeta/SentAnPod/logs"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Extend spaCy's NER model
def extend_ner_with_tesla(nlp):
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "ORG", "pattern": "Tesla"}]
    ruler.add_patterns(patterns)

# Load spaCy's pre-trained NER model
nlp = spacy.load("en_core_web_sm")
extend_ner_with_tesla(nlp)

# Function to extract assets (organizations/products) from text
def extract_assets(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]

# Load the emotion analysis model
emotion_analyzer = pipeline(
    "text-classification",
    model="/data/model/financial_emotion_model",
    tokenizer="/data/model/financial_emotion_model",
    device=0  # Use GPU
)

# Define label-to-emotion mapping
label_to_emotion = {
    "LABEL_0": "joy",
    "LABEL_1": "neutral",
    "LABEL_2": "fear"
}

# Process sentences and extract emotions
def process_file(file_path):
    # Read the content of the file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Extract assets and sentences
    assets = extract_assets(text)
    sentences = [sent.text.strip() for sent in nlp(text).sents]

    # Create a list to store results
    results = []
    for asset in assets:
        for sentence in sentences:
            if asset in sentence:
                # Analyze emotion for the sentence
                raw_prediction = emotion_analyzer(sentence)
                mapped_emotion = label_to_emotion[raw_prediction[0]["label"]]
                results.append({"phrase": sentence, "asset": asset, "emotion": mapped_emotion})
    return results

# Process all .txt files in the folder
def process_all_files(input_folder):
    dataset = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                logging.info(f"Processing file: {file_path}")
                try:
                    dataset.extend(process_file(file_path))
                    logging.info(f"Successfully processed file: {file_path}")
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")
    return dataset

# Main function
def main(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "podcast_asset_emotion_dataset.csv")

    # Process all files and create a DataFrame
    data = process_all_files(input_folder)
    df = pd.DataFrame(data)

    # Save the dataset to a CSV file
    df.to_csv(output_file, index=False)
    logging.info(f"Dataset saved to {output_file}")

# Input and output paths
input_folder = "/data/podcasts"
output_folder = "/data/dataset"

# Run the main function
main(input_folder, output_folder)
