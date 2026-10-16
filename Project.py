import random
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

warnings.filterwarnings('ignore')


data = {
    "greeting": [
        "hi", "hello", "hey there", "good morning", "good evening",
        "good afternoon", "hello there", "hey", "hi there", "howdy",
        "yo", "greetings", "hey bot", "hello friend", "hiya",
        "morning", "hello again", "hi how are you",
    ],
    "goodbye": [
        "bye", "goodbye", "see you later", "talk to you later",
        "I have to go", "catch you later", "see you soon",
        "have a nice day bye", "bye bye", "farewell",
        "I am leaving now", "good night", "see you tomorrow",
        "gotta run", "until next time", "take care bye",
    ],
    "thanks": [
        "thanks", "thank you", "thanks a lot", "I appreciate it",
        "that was helpful thank you", "many thanks", "thank you so much",
        "I am grateful", "thx", "thanks for your help", "cheers",
        "much appreciated", "thank you very much", "great thanks",
        "awesome thank you", "that helps thanks",
    ],
    "help": [
        "can you help me", "I need some help", "please help me",
        "could you help me with something", "I have a question",
        "can you explain this", "I need assistance", "will you help me",
        "help", "I am stuck", "can you assist me", "I need your help",
        "could you give me a hand", "I want to ask something",
        "can I ask you something", "help me please",
    ],
    "small_talk": [
        "how are you", "how is your day", "how are things", "what is up",
        "how do you feel", "is everything okay", "how is it going",
        "what are you doing", "how have you been",
        "are you having a good day", "what is new", "sup",
        "how is life", "are you okay", "how was your day",
        "what is going on",
    ],
    "identity": [
        "what is your name", "who are you", "tell me about yourself",
        "are you a robot", "are you a chatbot", "what should I call you",
        "what are you", "who am I talking to", "introduce yourself",
        "are you human", "who made you", "are you a real person",
        "what do you do", "tell me who you are", "do you have a name",
        "are you an AI",
    ],
    "unknown": [
        "what is the weather today", "how tall is mount everest",
        "what is the capital of France", "play some music",
        "book a flight to paris", "what time is it", "tell me a joke",
        "how do I cook pasta", "who won the football match",
        "translate this to french", "what is two plus two",
        "order me a pizza", "what is the stock price of apple",
        "how far is the moon", "recommend a movie",
        "set an alarm for seven",
    ],
}

## Creating DataFrame & Data set
row=[]

for intent,texts in data.items():
    for text in texts:
        row.append([text,intent])

Data=pd.DataFrame(row,columns=['text','intent'])

## Split Data 
x_train,x_test,y_train,y_test=train_test_split(Data['text'],Data['intent'],test_size=0.2,random_state=42)

## Freatures & Vectors
features=FeatureUnion([
    ('word',TfidfVectorizer(lowercase=True,ngram_range=(1,2))),
    ('char',TfidfVectorizer(lowercase=True,analyzer='char_wb',ngram_range=(2,4)))
])

## PIPLINE
pipeline=Pipeline([
    ('features',features),
    ('classifier',LogisticRegression(max_iter=2000,random_state=42))
])

## GridSearch
Search=GridSearchCV(
    pipeline,
    param_grid={'classifier__C':[1,10,100]},
    cv=5,
    scoring='accuracy'
)
Search.fit(x_train,y_train)
index=Search.best_index_
model=Search.best_estimator_
prediction=model.predict(x_test)

## Results & Train Report
print('Data Size:',len(Data))
print('Train Size:',len(x_train))
print('Test Size:',len(x_test))
print('Accuracy:',accuracy_score(y_test,prediction))
print('Report:',classification_report(y_test,prediction))



responses = {
    "greeting": [
        "Hello! How can I help you?",
        "Hi there! What can I do for you?",
        "Hey! Nice to talk to you.",
    ],
    "goodbye": [
        "Goodbye! Have a great day.",
        "See you later!",
        "Take care!",
    ],
    "thanks": [
        "You're welcome!",
        "Happy to help!",
        "No problem!",
    ],
    "help": [
        "Of course. What would you like help with?",
        "I'll do my best to help. Tell me more.",
        "Sure! Please explain what you need.",
    ],
    "small_talk": [
        "I'm doing well, thanks for asking! How can I help you?",
        "All good here! What would you like to talk about?",
    ],
    "identity": [
        "I'm a simple NLP chatbot built with Python.",
        "I'm an intent-based chatbot that uses TF-IDF and Logistic Regression.",
    ],
    "unknown": [
        "Sorry, I can't help with that yet. I only know greetings, thanks and small talk.",
        "I'm not sure I understand. Could you rephrase that?",
    ],
}

## defining Chatbot Function
def intent_and_confidence(massage):
    probilities=model.predict_proba([massage])[0]
    classes=model.named_steps['classifier'].classes_
    best_index=probilities.argmax()

    return classes[best_index] , probilities[best_index]


def chatbot_reply(massage,thershold=0.25):
    massage=massage.strip()

    if not massage:
        return 'how can i help you :)', None ,0.0

    intent,confidence=intent_and_confidence(massage)

    if confidence<thershold:
        intent='unknown'

    return random.choice(responses[intent]) , intent , confidence

## Conversation Loop
print('ChatBot is Ready!')
print("type 'quit' or 'exit' to stop")

while True:
    user_massage=input('YOU:')

    if user_massage.strip().lower() in {'quit','exit'}:
        print('BOT: Goodbye!')
        break

    reply,intent,confidence = chatbot_reply(user_massage)

    print(f'BOT :{reply}')

    if intent is not None:
        print(f'[Predicted Intent:{intent}] | [Predicted Confidence:{confidence:.2f}]')



    
    