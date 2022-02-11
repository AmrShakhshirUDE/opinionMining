#flask
from flask import Flask, jsonify, request, json,send_file,redirect,url_for,session
from flask_cors import CORS

#to import cnn model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import re
import numpy as np
from numpy import array
import pandas as pd
import pickle
import string
# from deep_translator import GoogleTranslator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



abbreviations ={
    "’": "'",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause'": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how does",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'am": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shalln't": "shall not",
    "shan't": "shall not",
    "shan't've": "shall not have",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "he is",
    "shouldn've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there'is": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "were't": "were not",
    " u ": "you",
    " ur ": "your",
    "you’re": "you are",
    "you're": "you are",
    " u'r ": "you are",
    " n ": "and",
    " lil mor ": "little more",
    " can' ": "cannot",
    " gov re ": "government regulations",
    " yea ": " yes ",
    " yeah ": " yes ",    
}

#translating emojis
with open('./dictionaries/Emoji_Dict.p', 'rb') as fp:
    Emoji_Dict = pickle.load(fp)
Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

def convert_emojis_to_word(text):
    for emot in Emoji_Dict:
        text = re.sub(r'('+emot+')', "_".join(Emoji_Dict[emot].replace(",","").replace(":","").split()) +' emojihasbeendetected ', text)
    return text

def cleanTweets(tweet):
    tweet= re.sub(r'@[A-Za-z0-9]+','',tweet)                            #remove @mentions;r-> the expression is a raw string,+ -> one char or more
    tweet= re.sub(r'#','',tweet)                                        #remove '#' Symbol
    tweet= re.sub(r'https?:\/\/\S+','websiteisgiven',tweet)             #replace hyper links
    translator = str.maketrans('', '', string.punctuation)
    tweet= tweet.translate(translator)                                  #remove punctuations
    return tweet

def getCleanTextIn(txt):
    if type(txt) is str:
        txt=txt.lower()
        for key in abbreviations:                                                   #transform abbreviations into complete words
            value = abbreviations[key]
            txt=txt.replace(key, value)
        return txt
    else:
        return txt

# loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    token = pickle.load(handle)

cnnGlove=load_model('cnnGlove') # load CNN model

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

'''flask configuration'''
app = Flask(__name__)
CORS(app)

@app.route('/predict/',methods=['POST'])
def predict():
    #Receive needed attributes
    tweet          = request.json['TWEET']
    print(tweet)
    inpuText=tweet
    text= convert_emojis_to_word(inpuText)
    text=cleanTweets(text)
    text=getCleanTextIn(text)

    word_list = text.split()
    number_of_words = len(word_list)

    if number_of_words < 3:
        message='Cannot predict, text should be more than 3 words!'
    elif (number_of_words <4 & has_numbers(text) ):
        message='Cannot predict, text should be more than 3 words excluding numbers!'
    else:
        encodedText = token.texts_to_sequences([text])
        maxLength = 100 # make input size equal model configuration size
        fixed = pad_sequences(encodedText,maxlen=maxLength,padding='pre') # could be configured as padding= 'post'

        '''predict subjectivity'''
        testPredict= (cnnGlove.predict(fixed) > 0.5).astype("int32")

        str(testPredict[0])

        subjectivity=""
        if testPredict[0] == 0:
            subjectivity='Objective'
            sub=0
        else:
            subjectivity='Subjective'
            sub=1

        message=subjectivity
    print(message)


    #Send result as JSON msg to frontend
    res = jsonify({'msg':message})    
    print (res)
    return res


@app.route('/')  #check connectivity
def connected():
    return'''
    <html>
        <h1>Connected to Subjectivity-Group Project Backend Side!!!</h1>
        <h4>Team members</h4>
        <ul>
            <li>Amr Shakhshir</li>
            <li>Sophia Abel</li>
            <li>Lena Greiner-Hiero</li>
            <li>Alina Krüger</li>
            <li>Chiara Loverso</li>
        </ul>
    </html>
    '''


if __name__ == '__main__':
    app.run(debug =True )