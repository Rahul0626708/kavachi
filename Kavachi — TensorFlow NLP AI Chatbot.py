import random
import json
import pickle
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import os


MODEL_FILE = "chatbot_model.h5"
DATA_FILE = "training_data.pkl"
INTENTS_FILE = "intents.json"
BOT_NAME = "Kavachi 🤖"



nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
    import train_chatbot  


model = load_model(MODEL_FILE)
data = pickle.load(open(DATA_FILE, 'rb'))
words = data['words']
classes = data['classes']
intents = data['data']


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_intent(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list):
    if not intents_list:
        return "I'm not sure I understand. Could you rephrase?"
    tag = intents_list[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure I understand."


def kavachi_response(message):
    intents_list = predict_intent(message)
    response = get_response(intents_list)
    return f"{BOT_NAME}: {response}"


def chat_fn(message, history):
    return kavachi_response(message)

demo = gr.ChatInterface(
    fn=chat_fn,
    title="💬 Kavachi — AI Chatbot",
    description="An AI-powered assistant built using TensorFlow, NLP, and data science.",
    theme="soft"
)

demo.launch()