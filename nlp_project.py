import re, nltk, os, unicodedata
import pickle
import inflect
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

# WordStemmer
st = LancasterStemmer()
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
        if " RT " in text[i].lower:
            clear = False
        if len(text[i]) < 20 and clear == True:
            clear = False
        if clear == True and compareEnglishWords(text[i]) == False:
            clear = False
        #Check with other tweets
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

# # Read CSV File
# cols = ['tweet_id', "airline_sentiment", "airline_sentiment_confidence", "negativereason", "negativereason_confidence", "airline", "airline_sentiment_gold", "name", "negativereason_gold", "retweet_count", "text", "tweet_coord", "tweet_created", "tweet_location", "user_timezone"]
# data_df = pd.read_csv(os.path.join("", 'Tweets.csv'), names=cols)

# # Get the required columns
# labels = data_df['airline_sentiment'].values
# text = data_df['text'].values

# # labels, text = filterTweets(labels, text)

# # Extract Training set
# trainLabels = labels[ 1 : int((len(labels)-1)*0.8) ]
# trainText = text[ 1 : int((len(text)-1)*0.8) ]

# # Extract Testing set
# testLabels = labels[ int((len(labels)-1)*0.8) :  len(labels)]
# testText = labels[ int((len(text)-1)*0.8) :  len(text)]

# # Tokenize Text Training Set and perform Case Folding
# Tokenized_text_training = []
# for i in range(0, len(trainText)):
#     removedUrls = remove_Urls(trainText[i])
#     temp = nltk.word_tokenize(removedUrls)
#     words = [w.lower() for w in temp]
#     words = normalizeText(words)
#     words = [st.stem(t) for t in words]
#     Tokenized_text_training.append(words)

# # Tokenize Text Testing Set and perform Case Folding
# Tokenized_text_testing = []
# for i in range(0, len(testText)):
#     removedUrls = remove_Urls(testText[i])
#     temp = nltk.word_tokenize(removedUrls)
#     words = [w.lower() for w in temp]
#     words = normalizeText(words)
#     words = [st.stem(t) for t in words]
#     Tokenized_text_testing.append(words)    

# with open('train', 'wb') as fp:
#     pickle.dump(Tokenized_text_training, fp)

# with open('test', 'wb') as fp:
#     pickle.dump(Tokenized_text_testing, fp)

with open ('train', 'rb') as fp:
    trainLoad = pickle.load(fp)

with open ('test', 'rb') as fp:
    testLoad = pickle.load(fp)

sentencesTrained = convertTokensToSentences(trainLoad)

vv = TfidfVectorizer(norm = None)
tfidf = vv.fit_transform(sentencesTrained)
print(sorted(vv.vocabulary_.items(), key=lambda x : x[1]))


# print(trainLoad[7])




