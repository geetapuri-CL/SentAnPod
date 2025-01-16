import os
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from transformers import pipeline
from datetime import datetime
import logging

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

# Load NASDAQ file and extract tech assets
def load_nasdaq_file(file_path):
    try:
        nasdaq_df = pd.read_excel(file_path)
        tech_assets = nasdaq_df["Company Name"].tolist()
        tech_assets = [name.split(",")[0] for name in tech_assets]  # Clean company names
        logging.info(f"Loaded {len(tech_assets)} tech companies from {file_path}")
        return tech_assets
    except Exception as e:
        logging.error(f"Error loading NASDAQ file: {e}")
        raise e

# Extend spaCy's NER with tech assets
def extend_ner_with_tech_assets(nlp, tech_assets):
    try:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [{"label": "ORG", "pattern": name} for name in tech_assets]
        ruler.add_patterns(patterns)
        logging.info("Extended NER with tech assets.")
    except Exception as e:
        logging.error(f"Error extending NER with tech assets: {e}")
        raise e

# Extract assets from text
def extract_assets(text, nlp):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

# Process sentences and analyze sentiment
def process_sentences(file_path, nlp, sentiment_pipeline):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        assets = extract_assets(text, nlp)
        sentences = [sent.text.strip() for sent in nlp(text).sents]

        results = []
        for asset in assets:
            for sentence in sentences:
                if asset in sentence:
                    sentiment = sentiment_pipeline(sentence)[0]
                    results.append({
                        "phrase": sentence,
                        "asset": asset,
                        "sentiment": sentiment["label"].lower(),
                        "score": sentiment["score"]
                    })
        logging.info(f"Successfully processed file: {file_path}")
        return results
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return []

# Process all files in the folder
def process_all_files(input_folder, nlp, sentiment_pipeline):
    dataset = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                logging.info(f"Processing file: {file_path}")
                dataset.extend(process_sentences(file_path, nlp, sentiment_pipeline))
    return dataset

# Main function
def main(nasdaq_file, input_folder, output_folder):
    try:
        # Load NASDAQ tech companies
        tech_assets = load_nasdaq_file(nasdaq_file)

        # Load spaCy model and extend NER
        nlp = spacy.load("en_core_web_sm")
        extend_ner_with_tech_assets(nlp, tech_assets)

        # Load FinBERT sentiment pipeline
        sentiment_pipeline = pipeline(
            "text-classification",
            model="yiyanghkust/finbert-tone",
            tokenizer="yiyanghkust/finbert-tone",
            device=0  # Use GPU
        )

        # Process all text files
        dataset = process_all_files(input_folder, nlp, sentiment_pipeline)

        # Save results
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "final_podcast_tech_sentiment_dataset.csv")
        df = pd.DataFrame(dataset)
        df.to_csv(output_file, index=False)
        logging.info(f"Final dataset saved to {output_file}")
    except Exception as e:
        logging.error(f"Error in main pipeline: {e}")
        raise e

# Input and output paths
nasdaq_file = "/data/NASDAQ_top30.xlsx"
input_folder = "/data/podcasts"
output_folder = "/data/dataset"

# Run the pipeline
main(nasdaq_file, input_folder, output_folder)
