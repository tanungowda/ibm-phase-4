import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("large_noisy_text_data.csv")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if isinstance(text, str):  # Ensure input is a string
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        words = word_tokenize(text)
        cleaned_text = " ".join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])
        return cleaned_text
    return ""

# Apply text cleaning
df["Cleaned_Text"] = df["Raw_Text"].apply(clean_text)

# Save cleaned dataset
df.to_csv("cleaned_large_noisy_text_data.csv", index=False)

print("Text cleaning completed. Cleaned dataset saved as cleaned_large_noisy_text_data.csv")
