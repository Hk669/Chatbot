import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()

#loading the data
intents = json.loads(open("intents.json").read())

#loading the previously made pkl files and trained model
words = pickle.load(open("words.pkl",'rb'))
classes = pickle.load(open("classes.pkl",'rb'))
model = load_model("chatbot_model.h5")

# it preprocess the sentences given by the user
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# converts sentences into bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    # this returns an array with len(words) times 0's
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # this replaces 1 with 0 if w is present in the words, else no changes i.e, 0
                bag[i] = 1
    return np.array(bag)


# it calls bagofwords and then the array of 1's and 0's is sent to the model for the prediction
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

# to get the response from the chatbot for question asked by the user
def getResponse(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            #if tag is present then it returns the random response given in the json file
            result = random.choice(i['responses'])
            break
    else:
        result = 'Sorry, I couldnt understand'
    return result

# API for the chatbot
from fastapi import FastAPI
from pydantic import BaseModel

#cretaed an instance of FastAPI as app
app = FastAPI()

# class for input message as string
class Message(BaseModel):
    message: str


# this block defines the endpoint
@app.post('/chatbot')
async def chatbot(message: Message):
    ints = predict_class(message.message)
    res = getResponse(ints, intents)
    return {'response': res}


# this block ensures that server is only run when code is executed directly
if __name__ == "__main__":

    # uvicorn is used to boost the performance 
    import uvicorn

    print("How can I help you?")
    uvicorn.run(app, host='127.0.0.1', port=5000)

