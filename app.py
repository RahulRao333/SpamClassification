import pickle

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)
model = pickle.load(open('Spam_model_1.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index1.html")


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [request.form.values()]

    prediction = Spam_model_1.predict(int_features)
    output = prediction[0]
    if output == 1:
        return render_template('index.html', prediction_text='Its a Spam Message')
    if output == 0:
        return render_template('index.html', prediction_text='It is not a Spam Message')


if __name__ == '__main__':
    app.run(debug=True)