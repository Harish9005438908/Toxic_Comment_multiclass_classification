# app.py
# This is the Python Flask backend for your application.
# It handles loading the model, processing user input, and making predictions.

import os
import re
import joblib
from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Create a Flask app instance
app = Flask(__name__)

# --- Load the model and vectorizer ---
# It's crucial to load these only once when the app starts.
# This saves time and memory.
print("Loading the model and vectorizer...")
try:
    model_path = os.path.join("./saved_models", 'best_toxic_model.joblib')
    vectorizer_path = os.path.join("./saved_models", 'tfidf_vectorizer.joblib')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure you have run the previous script and saved the files to the 'saved_models' folder.")
    exit()

# --- Preprocessing function (copied from previous script) ---
def preprocess_text(text):
    """
    Cleans and preprocesses a single text string.
    This function must be identical to the one used for training.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(filtered_words)

@app.route('/')
def home():
    """
    Renders the main HTML page of the application.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the web page.
    """
    # Get the data from the request
    data = request.get_json(force=True)
    comment = data['comment']

    # Preprocess the input comment
    processed_comment = preprocess_text(comment)

    # Transform the comment using the loaded TF-IDF vectorizer
    comment_vectorized = vectorizer.transform([processed_comment])

    # Make a prediction with the loaded model
    prediction = model.predict(comment_vectorized)[0]
    
    # Return the result as a JSON response
    result = "Toxic" if prediction == 1 else "Not Toxic"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    # You can change the port if needed. debug=True allows for live-reloading.
    app.run(port=5000, debug=True)