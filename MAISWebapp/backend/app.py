from flask import Flask, render_template, request
import joblib
import pickle
import os
# Import model
import model.main_wine_class as MWC
from model.main_wine_class import dummy
from model.main_wine_class import cat_to_price
#import torch

app = Flask(__name__)

# Instantiate models
svm_path = 'backend/model/results/svm.pickle'
#rfc_path = 'backend/model/results/rfc.pickle'
svm = pickle.load(open(svm_path, 'rb'))
#rfc = pickle.load(open(rfc_path, 'rb'))


#render the default webpage
@app.route('/')
def home():
    return render_template('index.html')

#redirect to success function
@app.route('/', methods=['POST'])
def make_prediction():
    text = request.form['description']
    print(len(text))
    if len(text) <= 2:
        return render_template('index.html', prediction="Please enter some text")

    prediction_text = MWC.process_input_text(str(text))
    svm_predicted = svm.predict(prediction_text)
    #rfc_predicted = rfc.predict(prediction_text)
    svm_prediction = cat_to_price[round(sum(svm_predicted)/len(svm_predicted))]
    #rfc_prediction = cat_to_price[round(sum(rfc_predicted)/len(rfc_predicted))]

    price_prediction = "The model predicts a price category of {}.".format(svm_prediction)
    return render_template('index.html', prediction=price_prediction)

if __name__ == '__main__':
    app.run(debug=True)
