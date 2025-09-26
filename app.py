from flask import Flask, request, render_template
from transformers import pipeline
import re
import os  
from flask import send_from_directory

app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True

# Specifying the pretrained sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.png', mimetype='image/png')

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove unwanted characters except basic punctuation
    text = re.sub(r'[^a-z0-9\s.!?]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sentiment_score_label(text):
    """Return signed sentiment score from HuggingFace pipeline"""
    result = sentiment_pipeline(text[:512])[0]  # truncate to first 512 tokens approx
    score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
    return score

def overall_sentiment_label(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Mixed"

def sliding_window_scores(scores, window_size=3):
    if not scores or len(scores) < window_size:
        return None
    max_sum = float('-inf')
    min_sum = float('inf')
    max_pos = (0, window_size - 1)
    min_pos = (0, window_size - 1)
    for i in range(len(scores) - window_size + 1):
        window_sum = sum(scores[i:i + window_size])
        if window_sum > max_sum:
            max_sum, max_pos = window_sum, (i, i + window_size - 1)
        if window_sum < min_sum:
            min_sum, min_pos = window_sum, (i, i + window_size - 1)
    return max_pos, min_pos

def kadane_algorithm(arr):
    if not arr:
        return None
    max_sum = cur_max = arr[0]
    start = end = s = 0
    for i in range(1, len(arr)):
        if cur_max + arr[i] < arr[i]:
            cur_max = arr[i]
            s = i
        else:
            cur_max += arr[i]
        if cur_max > max_sum:
            max_sum = cur_max
            start, end = s, i
    return start, end

def min_subarray(arr):
    if not arr:
        return None
    neg_arr = [-x for x in arr]
    return kadane_algorithm(neg_arr)

def word_break_all(s, word_dict):
    memo = {}
    def backtrack(idx):
        if idx == len(s):
            return ['']
        if idx in memo:
            return memo[idx]
        results = []
        for end in range(idx + 1, len(s) + 1):
            w = s[idx:end]
            if w.lower() in word_dict:
                suffixes = backtrack(end)
                for suf in suffixes:
                    combined = w + (' ' + suf if suf else '')
                    results.append(combined)
        memo[idx] = results
        return results
    return backtrack(0)

@app.route("/", methods=["GET","POST"])
def index():
    result = {}
    input_text = ""
    cleaned_text = ""
    overall_label = None
    compound_score = 0.0

    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        if input_text:
            cleaned_text = clean_text(input_text)  # Clean input here

            paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
            paragraphs_sentences = [
                re.split(r'(?<=[.!?]) +', p) if p else []
                for p in paragraphs
            ]

            compound_score = sentiment_score_label(cleaned_text)
            overall_label = overall_sentiment_label(compound_score)

            sentence_scores = []
            for para_sents in paragraphs_sentences:
                scores = [sentiment_score_label(s) for s in para_sents]
                sentence_scores.append(scores)

            flat_sentences = [s for para in paragraphs_sentences for s in para]
            flat_scores = [score for para in sentence_scores for score in para]

            most_pos_sentence, most_neg_sentence = "", ""
            if flat_scores:
                max_idx = flat_scores.index(max(flat_scores))
                min_idx = flat_scores.index(min(flat_scores))
                if max_idx < len(flat_sentences):
                    most_pos_sentence = flat_sentences[max_idx]
                if min_idx < len(flat_sentences):
                    most_neg_sentence = flat_sentences[min_idx]

            window_size = 3
            paragraph_windows = []
            for scores in sentence_scores:
                if len(scores) < window_size:
                    paragraph_windows.append(None)
                else:
                    paragraph_windows.append(sliding_window_scores(scores, window_size))

            paragraph_arbitrary = []
            for scores in sentence_scores:
                if not scores:
                    paragraph_arbitrary.append((None, None))
                else:
                    paragraph_arbitrary.append((
                        kadane_algorithm(scores),
                        min_subarray(scores)
                    ))

            word_dict = set(word.lower() for word in flat_sentences)

            word_break_results = []
            if paragraphs_sentences and paragraphs_sentences[0]:
                first_nospace = re.sub(r'\s+', '', paragraphs_sentences[0][0])
                word_break_results = word_break_all(first_nospace, word_dict)

            result = {
                "most_pos_sentence": most_pos_sentence,
                "most_neg_sentence": most_neg_sentence,
                "sentence_scores": list(zip(flat_sentences, flat_scores)),
                "compound_score": compound_score,
                "paragraph_windows": paragraph_windows,
                "paragraph_arbitrary": paragraph_arbitrary,
                "word_break_results": word_break_results,
            }

    return render_template("index.html", input_text=input_text, cleaned_text=cleaned_text, overall_label=overall_label, result=result)

if __name__ == "__main__":
    app.run(debug=True)
