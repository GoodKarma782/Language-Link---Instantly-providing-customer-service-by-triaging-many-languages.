# importing libraries
from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import os
import twilio
import twilio.rest
from twilio.rest import Client

le = LabelEncoder()

app = Flask(__name__)
@app.route('/')

def home():
    return render_template("home.html")

@app.route("/predict", methods = ["POST"])

def predict():
    # loading the dataset
    data = pd.read_csv("language_detection.csv")
    y = data["Language"]

    # label encoding
    y = le.fit_transform(y)

    #loading the model and cv
    model = pickle.load(open("model.pkl", "rb"))
    cv = pickle.load(open("transform.pkl", "rb"))

    if request.method == "POST":
        # taking the input
        text = request.form["text"]
        # preprocessing the text
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', '', text)
        text = re.sub(r'[[]]', '', text)
        text = text.lower()
        dat = [text]
        # creating the vector
        vect = cv.transform(dat).toarray()
        # prediction
        my_pred = model.predict(vect)
        my_pred = le.inverse_transform(my_pred)

    #TWILLIO
    account_sid = 'AC52432a3f6c5ffe025d402780e07e6813'
    auth_token = '8d239cca2de21bbb7831fa28cef289b0'
    client = Client(account_sid, auth_token)

    if my_pred[0] == "English":
        message = client.messages \
                        .create(
                            body = text,
                            from_ = '+16205221771',
                            to = '+918861615857'
                        )
        print(message.sid)

        return render_template("home.html", pred="Your query has been submitted. The above text detected is {}".format(my_pred[0]))

    if my_pred[0] == "Hindi":
        message = client.messages \
                        .create(
                            body = text,
                            from_ = '+16205221771',
                            to = '+918861615857'
                        )
        print(message.sid)

        return render_template("home.html", pred="आपकी क्वेरी सबमिट कर दी गई है। उपरोक्त पाठ है {}".format(my_pred[0]))

    if my_pred[0] == "Spanish":
        message = client.messages \
                        .create(
                            body = text,
                            from_ = '+16205221771',
                            to = '+918861615857'
                        )
        print(message.sid)

        return render_template("home.html", pred="Su consulta ha sido enviada. El texto anterior es {}".format(my_pred[0]))

    if my_pred[0] == "French":
        message = client.messages \
                        .create(
                            body = text,
                            from_ = '+16205221771',
                            to = '+918861615857'
                        )
        print(message.sid)

        return render_template("home.html", pred="Votre requête a été soumise. Le texte ci-dessus détecté est {}".format(my_pred[0]))

    if my_pred[0] == "Italian":
        message = client.messages \
                        .create(
                            body = text,
                            from_ = '+16205221771',
                            to = '+918861615857'
                        )
        print(message.sid)

        #RETURNING ON MAIN INDEX FILE
        return render_template("home.html", pred="La tua richiesta è stata inviata. Il testo sopra rilevato è {}".format(my_pred[0]))

if __name__ =="__main__":
    # app.run(debug=True)
    app.run(debug=True, port=5001)
