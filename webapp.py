import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv('https://raw.githubusercontent.com/Aneeshcoder/SPAM_Classification/main/SPAM%20text%20message.csv')
df.drop_duplicates(inplace=True)
df['Category'] = df['Category'].map({'ham':0,'spam':1})

def clean_data(message):
    message_without_punc = [character for character in message if character not in string.punctuation]
    message_without_punc = ''.join(message_without_punc)

    separator = ' '
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english')])

df['Message'] = df['Message'].apply(clean_data)

x = df['Message']
y = df['Category']

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
model = MultinomialNB().fit(x_train,y_train)
predictions = model.predict(x_test)

def predict(text):
    labels = ['NOT SPAM','SPAM']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message is looking to be '+labels[v])

st.title('SPAM Classifier')
user_input = st.text_input('Write your message')
submit = st.button('Predict')
if submit:
    answer = predict([user_input])
    st.text(answer)