
from flask import Flask, render_template, request, jsonify
import re
import math
import nltk
import pickle
import random
from nltk import tokenize
from stopword_list import stopword_list
from collections import Counter
from bs4 import BeautifulSoup
import requests
import re

# nltk.download('punkt')

app = Flask(__name__)

@app.route('/', methods=["GET"])
def home():
	return render_template("index.html")

@app.route('/api/v1/article/', methods=['POST'])
def process_api():
	kelas = []

	if request.method == 'POST':
		article = request.get_json(silent=True)
		kelas = classify(article)

	return jsonify(kelas)

@app.route('/api/v1/scrape/', methods=['POST'])
def scrape_api():
    kelas = []

    if request.method == 'POST':
        url_scrap = request.get_json(silent=True)
        content = requests.get(url_scrap)
        soup = BeautifulSoup(content.text, 'html.parser')
        soup.find_all(['p','b','div', 'h1'])
        for script in soup(["footer", "nav", "aside", "header", "ins", "style", "script", "a", "h2", "h3", "h4", "h5"]):
            script.decompose()
        string = soup.get_text()
        article = re.sub(r'("([^"]|"")*")', '', string)
        kelas = classify(article)

    return jsonify(kelas)


def classify(document):
    count = Counter()
    features = []

    # Load Model Train
    with open('model.data', 'rb') as filehandle:
        features = pickle.load(filehandle)
    
    document = document.lower()
    document = re.sub(r'([^a-zA-Z ]+?)', ' ', document)

    document = nltk.word_tokenize(document)

    document = [x for x in document if x not in stopword_list]

    for word in document:
        count[word] += 1
    
    anak_val   = math.log(1/3)
    remaja_val = math.log(1/3)
    dewasa_val = math.log(1/3)

    for feature in features:
        if feature == 'anakFeatures':
            for word in document:
                if word in features['anakFeatures']:
                    anak_val += features['anakFeatures'][word]
                else:
                    anak_val += features['smoothAnakFeatures']
                # print(anak)
        elif feature == 'remajaFeatures':
            for word in document:
                if word in features['remajaFeatures']:
                    remaja_val += features['remajaFeatures'][word]
                else:
                    remaja_val += features['smoothRemajaFeatures']
        elif feature == 'dewasaFeatures':
            for word in document:
                if word in features['dewasaFeatures']:
                    dewasa_val += features['dewasaFeatures'][word]
                else:
                    dewasa_val += features['smoothDewasaFeatures']

    if anak_val > remaja_val and anak_val > dewasa_val:
        return ('anak', anak_val, count)
    elif remaja_val > anak_val and remaja_val > dewasa_val:
        return ('remaja', remaja_val, count)
    elif dewasa_val > anak_val and dewasa_val > remaja_val:
        return ('dewasa', dewasa_val, count)
    else:
        return ('Tidak diketahui', anak_val)

if __name__ == '__main__':
    app.run()
