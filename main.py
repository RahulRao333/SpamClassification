# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
app = Flask(__name__)
modelvect=joblib.load('countvectorizer.pkl')
model=joblib.load('Multinomialspammodel.pkl')


@app.route("/")
def home():
    return render_template("index2.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        vect = modelvect.transform(data)
        my_prediction = model.predict(vect)
    output = my_prediction
    if output == [1]:
        return render_template('index2.html', prediction_text='Its a Spam Message')
    if output == [0]:
        return render_template('index2.html', prediction_text='Its not a Spam Message')


if __name__ == '__main__':
    app.run(debug=True)
