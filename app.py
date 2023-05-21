import csv
import json

import numpy as np

from model import preprocess_text, vectorizer
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    # Read questions from the CSV file
    questions = []
    with open('dataset.csv', 'r') as file:
        reader = csv.DictReader(file)
        seen_questions = set()
        for row in reader:
            question = row['question']
            if question not in seen_questions:
                seen_questions.add(question)
                questions.append(question)
                if len(questions) == 10:
                    break

    return render_template('index.html', questions=questions)


# Load the pickled model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the answers from the request
    answers = request.json['answers']

    # Preprocess the answers
    preprocessed_answers = [preprocess_text(answer) for answer in answers]

    # Transform the preprocessed answers using the same vectorizer used for training
    X_test = vectorizer.transform(preprocessed_answers)

    # Make predictions using the model
    results = []
    for answer, X_test_sample in zip(answers, X_test):
        reshaped_answer = X_test_sample.reshape(1, -1)  # Reshape the answer to a 2D array
        prediction = model.predict(reshaped_answer)  # Pass the reshaped answer
        prediction_list = prediction.tolist()  # Convert NumPy array to list
        results.append((answer, prediction_list))

    # Return the results as JSON
    return json.dumps(results)

    # Return the results as JSON
    return json.dumps(results)


# Define the route for rendering the HTML page

if __name__ == '__main__':
    app.run()
