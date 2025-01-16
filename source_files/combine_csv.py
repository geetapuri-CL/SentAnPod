import pandas as pd

# Load FinancialPhraseBank.csv
df_phrasebank = pd.read_csv("/data/FinancialPhraseBank.csv", names=["sentence", "sentiment", "emotion"], skiprows=1)

# Load financial_emotion_dataset.csv
df_emotion = pd.read_csv("/data/financial_emotion_dataset.csv", names=["sentence", "emotion"], skiprows=1)

# Check the datasets
print("FinancialPhraseBank:")
print(df_phrasebank.head())

print("\nFinancial Emotion Dataset:")
print(df_emotion.head())

# Drop the sentiment column
df_phrasebank.drop(columns=["sentiment"], inplace=True)

# Verify structure
print(df_phrasebank.head())
print(df_emotion.head())

# Combine the datasets
combined_df = pd.concat([df_phrasebank, df_emotion], ignore_index=True)

# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the combined dataset
print(f"Combined dataset size: {len(combined_df)}")
print(combined_df.head())

# Save the combined dataset
combined_df.to_csv("/data/CombinedFinancialEmotionDataset.csv", index=False)
print("Combined dataset saved to 'CombinedFinancialEmotionDataset.csv'.")
