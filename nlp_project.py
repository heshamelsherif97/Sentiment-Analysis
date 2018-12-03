import re, nltk, os, unicodedata
import pickle
import numpy as np
import inflect
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

st = LancasterStemmer()
nB = MultinomialNB()
neigh = KNeighborsClassifier()
rf = RandomForestClassifier(n_estimators = 100, random_state=0)

#List contating common english words
english_words = pd.read_csv(os.path.join("", 'english_words.csv'), names=[""])


def convertTokensToSentences(tokens):
    sentencesTrain = []
    for i in range(0, len(tokens)):
        temp = tokens[i]
        s = ""
        for j in range(0, len(temp)):
            s += temp[j]+ " "
        sentencesTrain.append(s)
    return sentencesTrain

def filterTweets(labels, text):
    filteredLabels = []
    filteredText = []
    for i in range(1, len(labels)):
        clear = True
        if " RT " in text[i]:
            clear = False
        if clear == True and len(text[i]) < 20:
            clear = False
        #Check with other tweets and with english common words
        if clear == True:
            filteredLabels.append(labels[i])
            filteredText.append(text[i])
    return filteredLabels, filteredText 


def compareEnglishWords(sentence):
    temp = nltk.word_tokenize(sentence)
    counter = 0
    for i in range(0, len(temp)):
        if temp[i] in english_words:
            counter = counter + 1
    percentage = counter / len(temp)
    if percentage > 0.15:
        return True
    else:
        return False


def remove_Urls(sentence):
    newWord = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', sentence, flags=re.MULTILINE)
    return newWord

def remove_punctuation(words):
    newWords = []
    for word in words:
        newWord = re.sub(r'[^\w\s]', '', word)
        if (newWord != ''):
            newWords.append(newWord)
    return newWords 

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    e = inflect.engine()
    newWords = []
    for word in words:
        if word.isdigit():
            newWord = e.number_to_words(word)
            newWords.append(newWord)
        else:
            newWords.append(word)
    return newWords

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    newWords = []
    for word in words:
        newWord = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        newWords.append(newWord)
    return newWords

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    newWords = []
    for word in words:
        if word not in stopwords.words('english'):
            newWords.append(word)
    return newWords

def normalizeText(words):
    g2 = remove_non_ascii(words)
    g3 = replace_numbers(g2)
    g4 = remove_punctuation(g3)
    g5 = remove_stopwords(g4)
    return g5

def Tokenize_text(text):
    Tokenized_text = []
    for i in range(0, len(text)):
        removedUrls = remove_Urls(text[i])
        temp = nltk.word_tokenize(removedUrls)
        words = [w.lower() for w in temp]
        words = normalizeText(words)
        words = [st.stem(t) for t in words]
        Tokenized_text.append(words)
    return Tokenized_text

def MultinomialNB_Classifier(training2sentences, text2sentences, trainLabels):
    vv = TfidfVectorizer(norm = None)
    tfidf = vv.fit_transform(training2sentences)
    nB.fit(tfidf, trainLabels)
    tfidf2 = vv.transform(text2sentences)
    return nB.predict(tfidf2)

def randomForest_Classifier(training2sentences, text2sentences, trainLabels):
    vv = TfidfVectorizer(norm = None)
    tfidf = vv.fit_transform(training2sentences)
    rf.fit(tfidf, trainLabels)
    tfidf2 = vv.transform(text2sentences)
    return rf.predict(tfidf2)

def knearstNeighbour_Classifier(training2sentences, text2sentences, trainLabels):
    vv = TfidfVectorizer(norm = None)
    tfidf = vv.fit_transform(training2sentences)
    neigh.fit(tfidf, trainLabels)
    tfidf2 = vv.transform(text2sentences)
    return neigh.predict(tfidf2)

def evaluateResult(realLabels, predictedLabels):
    return round(f1_score(realLabels, predictedLabels, average='micro') * 100, 2)

# Read CSV File
cols = ['tweet_id', "airline_sentiment", "airline_sentiment_confidence", "negativereason", "negativereason_confidence", "airline", "airline_sentiment_gold", "name", "negativereason_gold", "retweet_count", "text", "tweet_coord", "tweet_created", "tweet_location", "user_timezone"]
data_df = pd.read_csv(os.path.join("", 'Tweets.csv'), names=cols)

# Get the required columns
labels = data_df['airline_sentiment'].values
text = data_df['text'].values

# Filter Tweets
filtered_labels, filtered_text = filterTweets(labels, text)

# Extract Training set
trainLabels = labels[ 1 : int((len(labels)-1)*0.8) ]
trainText = text[ 1 : int((len(text)-1)*0.8) ]

# Extract filtered Training set 
filtered_trainLabels = filtered_labels[ 1 : int((len(filtered_labels)-1)*0.8) ]
filtered_trainText = filtered_text[ 1 : int((len(filtered_text)-1)*0.8) ]

# Extract Testing set
testLabels = labels[ int((len(labels)-1)*0.8) :  len(labels)]
testText = text[ int((len(text)-1)*0.8) :  len(text)]

# Extract filtered Testing set
filtered_testLabels = filtered_labels[ int((len(filtered_labels)-1)*0.8) :  len(filtered_labels)]
filtered_testText = filtered_text[ int((len(filtered_text)-1)*0.8) :  len(filtered_text)]

# Tokenize and normailize
text_training = Tokenize_text(trainText)
text_test = Tokenize_text(testText)

# convert tokens into sentences
training2sentences = convertTokensToSentences(text_training)
text2sentences = convertTokensToSentences(text_test)

# Tokenize and normailize filtered
filtered_text_training = Tokenize_text(filtered_trainText)
filtered_text_test = Tokenize_text(filtered_testText)

# convert tokens into sentences filtered
filtered_training2sentences = convertTokensToSentences(filtered_text_training)
filtered_text2sentences = convertTokensToSentences(filtered_text_test)

# Train and predict using classifiers
multiNomial = MultinomialNB_Classifier(training2sentences, text2sentences, trainLabels)
knearst = knearstNeighbour_Classifier(training2sentences, text2sentences, trainLabels)
randomForest = randomForest_Classifier(training2sentences, text2sentences, trainLabels)

# Train and predict using classifiers of filtered
filtered_multiNomial = MultinomialNB_Classifier(filtered_training2sentences, filtered_text2sentences, filtered_trainLabels)
filtered_knearst = knearstNeighbour_Classifier(filtered_training2sentences, filtered_text2sentences, filtered_trainLabels)
filtered_randomForest = randomForest_Classifier(filtered_training2sentences, filtered_text2sentences, filtered_trainLabels)

print("Before Filter")
print("---------------")
print("Multinomial Naive Bayes F1 Score")
print(str(evaluateResult(testLabels, multiNomial)) + "%")

print("K nearset Neigbour F1 Score")
print(str(evaluateResult(testLabels, knearst)) + "%")

print("Random Forest F1 Score")
print(str(evaluateResult(testLabels, randomForest)) + "%")
print("_________________________")

print("After Filter")
print("---------------")
print("Multinomial Naive Bayes F1 Score")
print(str(evaluateResult(filtered_testLabels, filtered_multiNomial)) + "%")

print("K nearset Neigbour F1 Score")
print(str(evaluateResult(filtered_testLabels, filtered_knearst)) + "%")

print("Random Forest F1 Score")
print(str(evaluateResult(filtered_testLabels, filtered_randomForest)) + "%")
print("_________________________")