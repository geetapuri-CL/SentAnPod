from transformers import pipeline

# Load the fine-tuned model
emotion_analyzer = pipeline(
    "text-classification", 
    model="/data/model/financial_emotion_model", 
    tokenizer="/data/model/financial_emotion_model",
    device=0 #use GPU
    )

# Predict emotion for new sentences
sentences = ["Tesla's stock is plummeting due to market conditions.", 
             "Apple's earnings exceeded expectations."]
             
predictions = emotion_analyzer(sentences)

# Define label-to-emotion mapping
label_to_emotion = {
    "LABEL_0": "joy",
    "LABEL_1": "neutral",
    "LABEL_2": "fear"
}

# Map predictions to meaningful labels
for sentence, prediction in zip(sentences, predictions):
    emotion = label_to_emotion[prediction["label"]]
    confidence = prediction["score"]
    print(f"Sentence: {sentence}")
    print(f"Emotion: {emotion} (Confidence: {confidence:.2f})")