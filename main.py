import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    if pd.isnull(text):  # Check for missing values
        return ''
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Initialize lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Lemmatize and stem tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]

    # Join tokens back into a string
    preprocessed_text = ' '.join(stemmed_tokens)

    return preprocessed_text


# Read data from Excel file
df = pd.read_excel('Team 3 - Complaints.xlsx')  # Replace 'Team 3 - Complaints.xlsx' with your file name

# Preprocess text column
df['preprocessed_text'] = df['narrative'].apply(preprocess_text)  # Replace 'narrative' with your column name

# Save preprocessed data to a new Excel file
df.to_excel('preprocessed_complaints.xlsx',
            index=False)  # Replace 'preprocessed_complaints.xlsx' with your desired output file name

print("Preprocessing completed. Preprocessed data saved to 'preprocessed_complaints.xlsx'.")