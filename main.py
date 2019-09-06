import pandas as pandas
from flask import Flask, render_template, request, url_for
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
plt.rc('font', size=14)

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
    # data manip
    data = pd.read_csv('MLTestAll.csv', header=0)
    data.drop(['Total Funding Amount'], axis=1, inplace=True)
    data.drop(['Number of Lead Investors'], axis=1, inplace=True)
    data.drop(['Number of Funding Rounds'], axis=1, inplace=True)
    data.fillna(0, inplace=True)
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]

    plt.hist(data['Number of Founders'])
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    figure = base64.b64encode(buffer.getbuffer()).decode('ascii')

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

    return render_template('results.html', prediction="{0:.0f}%".format(test_proba * 100),figure=figure)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
