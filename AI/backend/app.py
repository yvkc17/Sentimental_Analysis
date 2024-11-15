from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
import cohere

app = Flask(__name__)
CORS(app)

API_TOKEN = "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"
COHERE_API_KEY = "9j4ENdOLLz6eGv7YWtbOAWyLUUdjWWXzgBBVQbzU"

headers = {"Authorization": f"Bearer {API_TOKEN}"}
SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
co = cohere.ClientV2(api_key=COHERE_API_KEY)

label_mapping = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}

def query_sentiment(review):
    response = requests.post(SENTIMENT_API_URL, headers=headers, json={"inputs": review})
    return response.json()

def classify_sentiment(review):
    for attempt in range(5):
        result = query_sentiment(review)
        if isinstance(result, dict) and 'error' in result:
            if "loading" in result['error']:
                time.sleep(attempt + 1)
                continue

        if isinstance(result, list) and isinstance(result[0], list):
            sentiment_label = result[0][0]['label']
            confidence = result[0][0]['score']
            sentiment = label_mapping.get(sentiment_label, sentiment_label)
            sentiment_score = confidence if sentiment == "POSITIVE" else -confidence

            if sentiment == 'POSITIVE' and confidence < 0.7:
                sentiment = 'NEUTRAL'

            return review, sentiment, sentiment_score, confidence

    return review, "ERROR", 0, "Max retries exceeded"

def generate_summary_with_cohere(text):
    message = f"Generate a concise summary of these given reviews.\n{text}"
    response = co.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": message}]
    )
    return response.message.content[0].text

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    input_method = data.get('inputMethod')
    reviews = []

    if input_method == 'text':
        reviews.append(data.get('reviewText'))
    elif input_method == 'file':
        file_content = data.get('fileContent')
        reviews = file_content.splitlines()

    results = []
    sentiment_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
    overall_sentiment_score = 0

    for review in reviews:
        review_text, sentiment, sentiment_score, confidence = classify_sentiment(review)
        if sentiment != "ERROR":
            results.append([review_text, sentiment, sentiment_score])
            sentiment_counts[sentiment] += 1
            overall_sentiment_score += sentiment_score

    total_reviews = len(reviews)
    average_sentiment_score = overall_sentiment_score / total_reviews if total_reviews else 0
    overall_sentiment = "Neutral"
    if average_sentiment_score > 0:
        overall_sentiment = "Positive"
    elif average_sentiment_score < 0:
        overall_sentiment = "Negative"

    summary = generate_summary_with_cohere(" ".join([review[0] for review in results])) if results else "No summary available."

    response = {
        "sentiments": results,
        "counts": sentiment_counts,
        "overallSentiment": overall_sentiment,
        "summary": summary
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
