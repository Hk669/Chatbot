import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

# libraries for building the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation 
from tensorflow.keras.optimizers.legacy import SGD


lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent["tag"]))
            # add to our classes list
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))


pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
     bag=[]
     word_patterns = doc[0]
     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
     for word in words:
          bag.append(1) if word in word_patterns else bag.append(0)


     output_row = list(output_empty)
     output_row[classes.index(doc[1])] = 1

     training.append([bag, output_row])

# shuffling the training data to avoid underfitting or overfitting of the modela
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:,0])
train_y = list(training[:,1])

# model building
model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation='softmax'))

# Stochastic Gradient Descent -> optimizer
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# training the model
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)
model.save('chatbot_model.h5',history)
print("Done")