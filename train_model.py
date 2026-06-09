import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Input text and labels
X = data["text"]
y = data["personality"]

# Convert text into numbers
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vector, y)

# Save trained model
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Model trained successfully")