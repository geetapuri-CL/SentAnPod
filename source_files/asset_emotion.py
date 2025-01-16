import spacy
from transformers import pipeline
from spacy.pipeline import EntityRuler

# Function to extend spaCy's NER model with custom entities
def extend_ner_with_tesla(nlp):
    # Add EntityRuler to the pipeline
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    # Define custom patterns
    patterns = [{"label": "ORG", "pattern": "Tesla"}]
    # Add patterns to the ruler
    ruler.add_patterns(patterns)

# Load spaCy's pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Extend spaCy's NER with custom patterns
extend_ner_with_tesla(nlp)

# Function to extract assets (organizations/products) from text
def extract_assets(text):
    # Process the text
    doc = nlp(text)
    # Extract entities labeled as ORGANIZATION or PRODUCT
    assets = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    return assets

# Load an emotion analysis model
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

# Function to get emotions associated with assets
def get_asset_emotions(text):
    # Extract assets
    assets = extract_assets(text)
    asset_emotions = {}

    # Process sentences containing each asset
    sentences = text.split(".")
    for asset in assets:
        asset_emotions[asset] = []
        for sentence in sentences:
            if asset in sentence:
                # Analyze emotion for the sentence
                emotion = emotion_analyzer(sentence)
                # Map the label to its emotion
                mapped_emotion = label_to_emotion[emotion[0]['label']]
                asset_emotions[asset].append(mapped_emotion)
    return asset_emotions

# Example text
text = "Apple's stocks surged today, while Microsoft faced backlash due to layoffs. Tesla is also showing signs of volatility."

# Test the extraction and emotion analysis
print("Extracted Assets:", extract_assets(text))  # Should include Tesla
print("Asset Emotions:", get_asset_emotions(text))
