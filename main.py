import pandas as pandas
from flask import Flask, render_template, request, url_for
from joblib import load

# EDA Packages
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/",methods=['POST'])
def MLModel():
    model = load('crunchbase_ml_test.joblib')
    if request.method == 'POST':
        number_of_articles_in = request.form['number_of_articles']
        number_of_articles = int(number_of_articles_in)
        number_of_founders_in = request.form['number_of_founders']
        number_of_founders = int(number_of_founders_in)
        number_of_investors_in = request.form['number_of_investors']
        number_of_investors = int(number_of_investors_in)
        number_of_acquisitions_in = request.form['number_of_acquisitions']
        number_of_acquisitions = int(number_of_acquisitions_in)
        sample = np.array([(number_of_articles, number_of_founders, number_of_investors, number_of_acquisitions)])
        test_proba = model.predict_proba(sample)[0][1]



    return render_template('results.html', prediction="{0:.0f}%".format(test_proba * 100))


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
