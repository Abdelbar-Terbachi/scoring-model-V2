import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from nltk.corpus import stopwords
import string

# Load the dataset
dataset = pd.read_csv("dataset.csv", encoding="utf-8")

# Define Arabic stopwords and punctuation
arabic_stopwords = set(stopwords.words('arabic'))
arabic_punctuation = set(string.punctuation)


# Preprocess the data
def preprocess_text(text):
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in arabic_stopwords)

    # Remove punctuation
    text = ''.join(char for char in text if char not in arabic_punctuation)

    # Remove extra spaces
    text = re.sub(' +', ' ', text)

    # Convert to lowercase
    text = text.lower()

    return text


# Apply preprocessing to the 'answer' column
dataset['processed_answer'] = dataset['answer'].apply(preprocess_text)

# Prepare input data
X = dataset['processed_answer']
y = dataset['score']

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train the model
model = SVR(kernel='linear')
model.fit(X, y)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
