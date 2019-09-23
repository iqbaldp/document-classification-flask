from flask import Flask, render_template, request, jsonify
import re
import math
import nltk
import pickle
import random
from nltk import tokenize
from stopword_list import stopword_list

nltk.download('punkt')

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


def classify(document):
    features = []

    # Load Model Train
    with open('model.data', 'rb') as filehandle:
        features = pickle.load(filehandle)
    
    document = document.lower()
    document = re.sub(r'([^a-zA-Z ]+?)', ' ', document)

    for stop in stopword_list:
            if stop in document:
                document = document.replace(stop, '')

    document = nltk.word_tokenize(document)
    
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
        return ('anak', anak_val)
    elif remaja_val > anak_val and remaja_val > dewasa_val:
        return ('remaja', remaja_val)
    elif dewasa_val > anak_val and dewasa_val > remaja_val:
        return ('dewasa', dewasa_val)
    else:
        return ('gatauw', anak_val)

if __name__ == '__main__':
    app.run()
