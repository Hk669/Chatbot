# Chatbot Description
This chatbot is a machine learning-based conversational agent trained to understand and respond to user input. It utilizes a neural network model built with Keras, a deep learning library, to provide intelligent responses in natural language.

The chatbot's functionality is based on a bag-of-words approach, where input text is processed and transformed into a binary representation of word occurrence. The training data consists of a collection of documents, with each document containing a list of words or tokens.

During the training process, the chatbot preprocesses the documents by applying lemmatization and lowercasing to the words. It then constructs a bag-of-words representation for each document by encoding the presence or absence of words from a predefined vocabulary.

The training data is split into input (bag-of-words) and output representations. The input is fed into a sequential neural network model, which consists of multiple dense layers with dropout regularization. The model is compiled with a categorical cross-entropy loss function and optimized using the stochastic gradient descent (SGD) algorithm.

The trained model can be used to generate responses by inputting new text data. The chatbot selects the most likely class or category for the input based on the model's output probabilities. It then formulates a response based on predefined mappings or patterns associated with the predicted class.

The chatbot's performance can be improved by training it on a diverse and representative dataset, fine-tuning the model architecture, and optimizing the hyperparameters.

# Usage
To use the chatbot, you need to provide a set of training documents in the appropriate format. Each document should consist of a list of words or tokens along with the corresponding class or category.

Once the training data is prepared, run the training script to train the model on the data. After training, the model can be saved for future use.

To interact with the chatbot, provide user input as text data. The chatbot will process the input, generate a response based on the trained model, and display it to the user.

# Sample of the bot

<img width="364" alt="image" src="https://github.com/Hk669/Chatbot/assets/96101829/9039b045-109d-49bc-9fda-92b0b56622bb">

The input is given to the bot and it understands the sentence and returns the relevant answer which is trained to the model 

<img width="397" alt="image" src="https://github.com/Hk669/Chatbot/assets/96101829/19fd0616-d522-4b96-b533-256543f95a58">
