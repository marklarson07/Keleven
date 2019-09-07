import pandas as pd
from flask import Flask, render_template, request, url_for
from joblib import load
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import base64

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


        # data manip
        data = pd.read_csv('MLTestAll.csv', header=0)
        data.drop(['Total Funding Amount'], axis=1, inplace=True)
        data.drop(['Number of Lead Investors'], axis=1, inplace=True)
        data.drop(['Number of Funding Rounds'], axis=1, inplace=True)
        data.fillna(0, inplace=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)

        # num articles
        ax1.hist(data['Number of Articles'], rwidth=.75)
        ax1.set_title('Number of Articles')
        ax1.set_xlabel('Number of Articles')
        ax1.set_ylabel('Count')
        ax1.axvline(x=number_of_articles, color='r', linestyle='solid', linewidth=2)

        # num founders
        ax2.hist(data['Number of Founders'], rwidth=.75)
        ax2.set_title('Number of Founders')
        ax2.set_xlabel('Number of Founders')
        ax2.set_ylabel('Count')
        ax2.axvline(x=number_of_founders, color='r', linestyle='solid', linewidth=2)

        # num investors
        ax3.hist(data['Number of Investors'], rwidth=.75)
        ax3.set_title('Number of Investors')
        ax3.set_xlabel('Number of Investors')
        ax3.set_ylabel('Count')
        ax3.axvline(x=number_of_investors, color='r', linestyle='solid', linewidth=2)

        # num acquisitions
        ax4.hist(data['Number of Acquisitions'], rwidth=.75)
        ax4.set_title('Number of Acquisitions')
        ax4.set_xlabel('Number of Acquisitions')
        ax4.set_ylabel('Count')
        ax4.axvline(x=number_of_acquisitions, color='r', linestyle='solid', linewidth=2)
        fig.tight_layout()
        # plt.show()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        figure = base64.b64encode(buffer.getbuffer()).decode('ascii')

    return render_template('results.html', prediction="{0:.0f}%".format(test_proba * 100),figure=figure)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
