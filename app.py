from flask import Flask, render_template, request, flash, redirect
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

app = Flask(__name__)
comments = pd.read_csv("C:\\Users\\Bhavana Padala\\Documents\\Jupyter Notebook\\Toxic Text Classification\\cleaned_data.csv")
X_train, X_test, y_train, y_test = train_test_split(comments['comment_text'], comments['toxic'], test_size=0.2, random_state=42)

vect =  TfidfVectorizer( lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1),dtype=np.float32)
tfidf = vect.fit(X_train.values.astype('U'))
toxic_model = pickle.load(open("toxic_model.pkl", "rb"))
severe_toxic_model = pickle.load(open("severe_toxic_model.pkl", "rb"))
obscene_model = pickle.load(open("obscene_model.pkl", "rb"))
insult_model = pickle.load(open("insult_model.pkl", "rb"))
threat_model = pickle.load(open("threat_model.pkl", "rb"))
identity_hate_model = pickle.load(open("identity_hate_model.pkl", "rb"))
def get_proba(n):
    a = np.argmax(n,axis=1)
    print(n)
    print(a)
    res = n[0][a]
    return res
def predict_toxicity(sentence):
    ss = pd.Series(sentence)
    sentence = tfidf.transform(ss)

    toxic_probability = toxic_model.predict_proba(sentence)
    toxic_probability  = get_proba(toxic_probability)
    toxic_ = toxic_model.predict(sentence)

    severe_toxic_probability = severe_toxic_model.predict_proba(sentence)
    severe_toxic_probability = get_proba(severe_toxic_probability)
    severe_toxic_ = severe_toxic_model.predict(sentence)

    obscene_probability= obscene_model.predict_proba(sentence)
    obscene_probability = get_proba(obscene_probability)
    obscene_ = obscene_model.predict(sentence)

    insult_probability = insult_model.predict_proba(sentence)
    insult_probability = get_proba(insult_probability)
    insult_ = insult_model.predict(sentence)

    threat_probability = threat_model.predict_proba(sentence)
    threat_probability = get_proba(threat_probability)
    threat_ = threat_model.predict(sentence)

    identity_hate_probability = identity_hate_model.predict_proba(sentence)
    identity_hate_probability = get_proba(identity_hate_probability)
    identity_hate_ = identity_hate_model.predict(sentence)

    return toxic_probability,toxic_,severe_toxic_probability,severe_toxic_,obscene_probability,obscene_,insult_probability,insult_,threat_probability,threat_,identity_hate_probability,identity_hate_

@app.route('/', methods = ['GET','POST'])
def home():
     return render_template("index.html")

@app.route('/result', methods = ['POST'])
def prediction():
    if request.method == 'POST':

        sentence = request.form['comment']

        toxic_probability,toxic_,severe_toxic_probability,severe_toxic_,obscene_probability,obscene_,insult_probability,insult_,threat_probability,threat_,identity_hate_probability,identity_hate_ = predict_toxicity(sentence)
        if toxic_ == 1 or severe_toxic_ == 1 or obscene_ == 1 or insult_ == 1 or threat_ == 1 or identity_hate_ == 1:
            toxicity = 1
            toxic_predict = "This comment's toxic probability is {}%.".format(round(np.max(toxic_probability)*100, 2))
            severe_toxic_predict = "This comment's severetoxic probability is {}%.".format(round(np.max(severe_toxic_probability)*100, 2))
            obscene_predict = "This comment's obscene  probability is {}%.".format(round(np.max(obscene_probability)*100, 2))
            insult_predict = "This comment's insult probability is {}%.".format(round(np.max(insult_probability)*100, 2))
            threat_predict = "This comment's threat probability is {}%.".format(round(np.max(threat_probability)*100, 2))
            identity_hate_predict = "This comment's identity hate probability is {}%.".format(round(np.max(identity_hate_probability)*100, 2))
            return render_template("result.html",toxicity = toxicity,toxic_predict=toxic_predict,severe_toxic_=severe_toxic_predict,obscene_predict=obscene_predict,insult_predict=insult_predict,threat_predict = threat_predict,identity_hate_predict = identity_hate_predict)
        else:
            predicts = 'This comment is perfectly alright'
            return render_template("result.html", predicts = predicts)
    # else:
    #     toxicity = False
    #     predicts = "Please enter relevant information."
    #     return render_template("result.html", toxicity = toxicity, predicts = predicts)
# app.route('/resultt', methods = ['POST','GET'])
# def resultt():
#     if request.method == 'POST':
#
#         sentence = request.form['comment']
#
#         toxic_probability,severe_toxic_probability,obscene_probability,insult_probability,threat_probability,identity_hate_probability = predict_toxicity(sentence)
#         if toxic_ == 1 or severe_toxic_ == 1 or obscene_ == 1 or insult_ == 1 or threat_ == 1 or identity_hate_ == 1:
#             toxicity = 1
#             toxic_predict = "This comment's toxic probability is {}%.".format(round(np.max(toxic_probability)*100, 2))
#             severe_toxic_predict = "This comment's toxic probability is {}%.".format(round(np.max(severe_toxic_probability)*100, 2))
#             obscene_predict = "This comment's toxic probability is {}%.".format(round(np.max(obscene_probability)*100, 2))
#             insult_predict = "This comment's toxic probability is {}%.".format(round(np.max(insult_probability)*100, 2))
#             threat_predict = "This comment's toxic probability is {}%.".format(round(np.max(threat_probability)*100, 2))
#             identity_hate_predict = "This comment's toxic probability is {}%.".format(round(np.max(identity_hate_probability)*100, 2))
#             return render_template("result.html",toxicity = toxicity,toxic_predict=toxic_predict,severe_toxic_=severe_toxic_predict,obscene_predict=obscene_predict,insult_predict=insult_predict,threat_predict = threat_predict,identity_hate_predict = identity_hate_predict)
#         else:
#             toxicity = 0
#             predicts = 'This comment is perfectly alright'
#             return render_template("result.html", toxicity = toxicity, predicts = predicts)
#     else:
#         toxicity = False
#         predicts = "Please enter relevant information."
#         return render_template("result.html", toxicity = toxicity, predicts = predicts)

if __name__ == '__main__':
    app.run(debug = True)
