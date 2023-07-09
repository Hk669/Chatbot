import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import load_model

#from flask import Flask, request, jsonify

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl",'rb'))
classes = pickle.load(open("classes.pkl",'rb'))
model = load_model("chatbot_model.h5")

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key = lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def getResponse(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = 'Sorry, I couldnt understand'
    return result

print("Bot started running!")

from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    message : str

@app.route('/chatbot')
async def chatbot(request: Request):
    data = await request.json()
    message = data['message']
    ints = predict_class(message)
    res = getResponse(ints, intents)
    return {'response': res}

if __name__ == "__main__":
    import uvicorn
    print("How can i help you")
    uvicorn.run(app, host='127.0.0.1',port = 5000)



#API function
'''app = Flask(__name__)

@app.route('/chatbot', methods=['GET','POST'])
def chat():
    data = request.get_json()
    message = data['message']
    ints = predict_class(message)
    res = getResponse(ints, intents)
    return jsonify({'response': res})

if __name__ == '__main__':
    print("Bot started running!")
    app.run(debug=True)'''
