import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd





# Read the Excel file
excel_file = pd.read_excel('preprocessed_complaints.xlsx')  # Replace "your_excel_file.xlsx" with the path to your Excel file

# Save as CSV
excel_file.to_csv("output.csv", index=False)  # Specify the path where you want to save the CSV file

# Load the dataset
data = pd.read_csv("output.csv")  # Replace "complaints_dataset.csv" with your dataset file path

data=data.dropna(subset=['preprocessed_text'])
# Split data into features (complaints) and target variable (complaint type)
X = data['preprocessed_text']
y = data['product']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features based on your data size
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test_tfidf)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))





