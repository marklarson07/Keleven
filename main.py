from flask import Flask, render_template, url_for, request
import numpy as np
from joblib import load

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/",methods=['POST'])
def MLModel():
    model = load('crunchbase_ml_test.joblib')
    if request.method == 'POST':
        number_of_articles_in = request.form['number_of_articles']
        number_of_articles = [number_of_articles_in]
        number_of_founders_in = request.form['number_of_founders']
        number_of_founders = [number_of_founders_in]
        number_of_investors_in = request.form['number_of_investors']
        number_of_investors = [number_of_investors_in]
        number_of_acquisitions_in = request.form['number_of_acquisitions']
        number_of_acquisitions = [number_of_acquisitions_in]
        sample = np.array([(number_of_articles, number_of_founders, number_of_investors, number_of_acquisitions)])
        test_proba = model.predict_proba(sample)


    return render_template('predict.html', prediction=test_proba)

if __name__ == "__main__":
    app.run(debug=True)






