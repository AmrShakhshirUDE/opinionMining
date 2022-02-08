from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.models import load_model

import re
import numpy as np
from numpy import array
import pandas as pd
import pickle
import string
from deep_translator import GoogleTranslator

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

indomain= pd.read_csv('./datasets/indomain.csv', encoding='utf8') #onlyTweets
outdomain= pd.read_csv('./datasets/outdomain.csv',sep=',', encoding='utf-8') #onlyTweets

#translating emojis
with open('./dictionaries/Emoji_Dict.p', 'rb') as fp:
    Emoji_Dict = pickle.load(fp)
Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

def convert_emojis_to_word(text):
    for emot in Emoji_Dict:
        text = re.sub(r'('+emot+')', "_".join(Emoji_Dict[emot].replace(",","").replace(":","").split()) +' emojihasbeendetected ', text)
    return text

indomain["tweet"]=indomain["tweet"].apply(convert_emojis_to_word)
outdomain["tweet"]=outdomain["tweet"].apply(convert_emojis_to_word)

def cleanTweets(tweet):
    tweet= re.sub(r'@[A-Za-z0-9]+','',tweet)                            #remove @mentions;r-> the expression is a raw string,+ -> one char or more
    tweet= re.sub(r'#','',tweet)                                        #remove '#' Symbol
    tweet= re.sub(r'https?:\/\/\S+','websiteisgiven',tweet)             #replace hyper links
    translator = str.maketrans('', '', string.punctuation)
    tweet= tweet.translate(translator)                                  #remove punctuations
    return tweet

indomain['tweet']= indomain['tweet'].apply(lambda txt:cleanTweets(txt))
outdomain['tweet']= outdomain['tweet'].apply(lambda txt:cleanTweets(txt))

inText=' '.join(indomain['tweet'])
inText=inText.split()
inFreqComm= pd.Series(inText).value_counts()
inRare=inFreqComm[inFreqComm.values==1] # get words that have been occured a single time to be removed from dataset

def getCleanTextIn(txt):
    if type(txt) is str:
        txt=txt.lower()
        for key in abbreviations:                                                   #transform abbreviations into complete words
            value = abbreviations[key]
            txt=txt.replace(key, value)
        return txt
    else:
        return txt

ouText=' '.join(outdomain['tweet'])
ouText=ouText.split()
outFreqComm= pd.Series(ouText).value_counts()
outRare=outFreqComm[outFreqComm.values==1] # get words that have been occured a single time to be removed from dataset

def getCleanTextOut(txt):
    if type(txt) is str:
        txt=txt.lower()
        for key in abbreviations:                                                   #transform abbreviations into complete words
            value = abbreviations[key]
            txt=txt.replace(key, value)
        return txt
    else:
        return txt

####BACK-TRANSLATION####
'''text-translation function'''
# from deep_translator import GoogleTranslator

# translated1=[]
# for i in indomain['tweet']:
#     translated1.append(GoogleTranslator(source='auto', target='de').translate(i))

# # to_translate = 'I want to translate this text'
# # translated = GoogleTranslator(source='auto', target='de').translate(to_translate)
# # outpout -> Ich möchte diesen Text übersetzen

# translated=[]
# for i in translated1:
#     translated.append(GoogleTranslator(source='auto', target='en').translate(i))

# tweets=indomain["tweet"].tolist()
# tweets=tweets+translated

'''write translated text to file'''
# textfile = open("./translated/listFlie.txt", "w",encoding="utf-8")
# for element in tweets:
#     textfile.write(element + "\n")
# textfile.close()

'''read translated text'''
#Make sure to remove last empty line from listFile.txt before run this snippet
my_file = open("./translated/listFlie.txt", "r",encoding="utf-8",newline="\r\n")
listText = my_file.read().split("\r\n")

'''extract only translated entries - (drop origin)'''
translatedList= pd.Series(listText)
start=len(indomain['tweet'])
translatedList=translatedList[start:len(translatedList)]

indomain['tweet']= indomain['tweet'].apply(lambda txt:getCleanTextIn(txt))      #for abbreviations
outdomain['tweet']= outdomain['tweet'].apply(lambda txt:getCleanTextOut(txt))   #for abbreviations
translatedList= translatedList.apply(lambda txt:cleanTweets(txt))               #text preprocessing
translatedList= translatedList.apply(lambda txt:getCleanTextIn(txt))            #for abbreviations

inText = indomain['tweet'].tolist()   #convert series to list
ouText = outdomain['tweet'].tolist()   #convert series to list
translatedList=translatedList.tolist()

'''concatinate origin & back-translated'''
flag=0
text1=inText

if flag ==0:
    text1= inText+translatedList
    flag+=1

text= text1

indomain["subjectivity"] = indomain["subjectivity"].astype('category')
target = indomain['subjectivity'].cat.codes
target = pd.concat([target]*2, ignore_index=True) #repeat target values *x times (add back-translated target)

outdomain["subjectivity"] = outdomain["actual"].astype('category')
ouTarget = outdomain['subjectivity'].cat.codes

'''tokeniz text -> encode tokenized words -> pad input to be all at the same size'''
token = Tokenizer()
token.fit_on_texts(text)

vocabSize = len(token.word_index) + 1 #4276 vocabularies/ with back-translated 4794

encodedText = token.texts_to_sequences(text)

maxLength = 100 # make all rows equal in size
fixed = pad_sequences(encodedText,maxlen=maxLength,padding='pre') # could be configured as padding= 'post'

'''GLOVE: Global Vectors'''
GloveVectors = dict()
file = open('dictionaries/gloveTwitter/glove.6B.300d.txt', encoding='utf-8')

for line in file:
    values = line.split()
    word = values[0]                    # to skip the first element which is a word 
    vectors = np.asarray(values[1:])    # to start from the numbers that are located after the word
    GloveVectors[word] = vectors
file.close()

wordVectorMatrix=np.zeros((vocabSize, 300)) # each word at vocab, should have 300 more dimentsion
currentWord=0
new_Words=0
for word, index in token.word_index.items():
    vector= GloveVectors.get(word)
    if vector is not None:
        currentWord += 1
        wordVectorMatrix[index] = vector
    else:
        new_Words +=1

#number of words found in GloVe:    3908    / back translated 4384
#number of words out of GloVe:      367     / back translated 409

'''Building CNN model with (Keras)'''
X_train, X_tset, Y_train, Y_test = train_test_split(fixed, target, random_state=42, test_size= 0.215, shuffle=True,stratify=target)

'''CNN model'''
vecSize = 300

model = Sequential()
model.add(Embedding(vocabSize, vecSize, input_length = maxLength, weights = [wordVectorMatrix], trainable = False))

model.add(Conv1D(50, 8, activation='relu'))
model.add(MaxPooling1D(5))

# model.add(Dense(32, activation='relu'))

# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.01))

model.add(GlobalMaxPooling1D())
model.add(Dropout(0.35))

model.add(Dense(1, activation='sigmoid'))
model.add(Dropout(0.55))

model.compile(optimizer=Adam(learning_rate=0.001), loss= 'binary_crossentropy', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, Y_train, epochs=32,batch_size=2, validation_data=(X_tset, Y_test))# best results on epochs = 160

cnnGlove=load_model('cnnGlove') # load best model so far

'''in-domain evaluation'''
testPredict= (model.predict(fixed) > 0.5).astype("int32")
testPredictSavedIn= (cnnGlove.predict(fixed) > 0.5).astype("int32")
print ('in-domain \n',classification_report(testPredict, target))
print ('in-domain Saved model \n',classification_report(testPredictSavedIn, target))

'''save model'''
model.save("cnnGlove2")

'''Evaluate on out-domain'''
encodedOuText = token.texts_to_sequences(ouText)
maxLength = 100 # make all rows equal in size
outFixed = pad_sequences(encodedOuText,maxlen=maxLength,padding='pre') # could be configured as padding= 'post'
testPredictOut= (model.predict(outFixed) > 0.5).astype("int32")
print ('out-domain \n',classification_report(testPredictOut, ouTarget))

testPredictSavedOut= (cnnGlove.predict(outFixed) > 0.5).astype("int32")
print ('out-domain Saved model \n',classification_report(testPredictSavedOut, ouTarget))

'''CLUSTERING PART'''
from matplotlib import pyplot as plt
# %matplotlib inline
import plotly.express as px
#Libraries for clustering
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizerTf = TfidfVectorizer(stop_words='english')
vectorizerNtf = TfidfVectorizer(stop_words='english',analyzer='word',ngram_range=(1,2))  
                                                                    #TF-IDF (n-gram level)// 
                                                                    #consider n-grams between 1 (individual words) 
                                                                    # and 2 (sequences of 2 words or bi-grams)
vectorizerBag = CountVectorizer(stop_words='english')

def cleanWebsite(tweet):
    tweet= re.sub(r'websiteisgiven','',tweet) #remove websiteisgiven ;r-> the expression is a raw string,+ -> one char or more
    return tweet

indomain['tweetClus']=indomain['tweet'].apply(lambda txt:cleanWebsite(txt))
textCluster=indomain['tweetClus'].tolist()

xTf=vectorizerTf.fit_transform(inText)
xBag=vectorizerBag.fit_transform(inText)
xNtf = vectorizerNtf.fit_transform(inText)

#Test increments of 100 clusters using elbow method
distortions = []
sse={}
K=range(1,10)
# for k in np.arange(100,900,100):
for k in K:
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(xTf)
    distortions.append(kmeans.inertia_)

plt.plot(K,distortions,'bx-')
plt.xlabel('Values for K')
plt.ylabel('SSE')
plt.title('The Elbow Method showing the optimal k')
plt.show()

'''Create clusters'''
k=8
kModel=KMeans(n_clusters=k,init='k-means++', max_iter=10000, n_init=1)
kModel.fit(xTf)

indomain['Cluster']=kModel.labels_

'''Output results to a text file'''
# clusters= indomain.groupby('Cluster')
# for cluster in clusters.groups:
#     f=open('Cluster'+ str(cluster)+'.csv','w') #create csv file
#     data=clusters.get_group(cluster)[['tweet','subjectivity']] #get tweet & subjectivity columns
#     f.write(data.to_csv(index_label='id')) #set index to id
#     f.close()

'''Understand what each cluster refers to'''
print('Cluster centroids: \n')
orderCentroids= kModel.cluster_centers_.argsort()[:,::-1]
terms=vectorizerTf.get_feature_names_out()

for i in range(k):
    print('Cluster %d:' %i)
    for j in orderCentroids[i, :10]: # print out 10 feature terms of each cluster
        print (' %s' %terms[j])
    print('-----------')

'''Evaluating Clustering method'''
from sklearn.metrics import silhouette_score
print('k-means evaluation at k=8: ',silhouette_score(xTf, labels=kModel.predict(xTf)))

'''DBSCAN-Cluster'''
from sklearn.cluster import DBSCAN
dbClus = DBSCAN(eps=1, min_samples=2).fit(xTf)

indomain['DbCluster']=dbClus.labels_
clustUniq=np.unique(dbClus.labels_)

print('Cluster centroids: \n')
# orderDbCentroids= dbClus.cluster_centers_.argsort()[:,::-1]
terms=vectorizerTf.get_feature_names_out()

for i in clustUniq:
    print('Cluster %d:' %i)
    for j in range(10): # print out 10 feature terms of each cluster
        print (' %s' %terms[j])
    print('-----------')

print('DBSCAN evaluation: ',silhouette_score(xTf, labels=dbClus.fit_predict(xTf)))