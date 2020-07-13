import json
import random
import numpy as np
from collections import OrderedDict
from collections import Counter
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer as st
from nltk.tag import pos_tag
from nltk.corpus import stopwords as stw

DATA_AMOUNT = 100000

def RemoveShorts(sentence):
    wordList = []
    for w in sentence:
        if len(w) > 2:
            wordList.append(w)

    return wordList
    
def RemoveStw(sentence):
    wordList = []
    for w in sentence:  
        if w not in stw.words('english') + ['thi']:
            wordList.append(w)

    return wordList

def NLP(data):
    newData = []
    for sentence in data:
        sentence = RegexpTokenizer('[\w]+').tokenize(sentence) # Tokenization
        sentence = [st().stem(w) for w in sentence] # Stemming
        sentence = RemoveStw(sentence) # Remove Stopwords
        sentence = RemoveShorts(sentence) # Remove Shortwords

        newData.append(sentence)
        print('Data {}/{} Preprocessed!'.format(len(newData), len(data)))
    
    return newData

def IntegerEncode(data):
    print('Summary word...')
    words = np.hstack(data)
    print('All words summaried')

    print('Analyze frequency...')
    vocab = Counter(words)
    vocab = vocab.most_common()
    print('Word frequency analyzed')

    encodedWords = OrderedDict()

    for i, word in enumerate(vocab):
        encodedWords[word[0]] = i + 1
        print('Word {}/{} Encoded!'.format(i, len(vocab)))

    return encodedWords

if __name__ == "__main__":    
    preprocessed = OrderedDict() # Preprocessed
    preprocessed['overall'] = []
    preprocessed['reviewText'] = []

    overallList, reviewTextList = [], []
    badOverallList, goodOverallList = [], []
    badReviewTextList, goodReviewTextList = [], []

    TEMP = []
    
    print('Data Loading...')
    data = [json.loads(line) for line in open('Grocery_and_Gourmet_Food.json', 'r')] # Original
    print('Data Loaded!')

    for i in range(len(data)):
        try:
            data[i]['reviewText']
        except KeyError:
            print('Data {}/{} is empty.'.format(i, DATA_AMOUNT))
        else:
            if data[i]['overall'] == 1.0 and len(badOverallList) < DATA_AMOUNT:
                badOverallList.append(data[i]['overall'])
                badReviewTextList.append(data[i]['reviewText'])
                print('Data {}/{} Loaded!(1.0)'.format(len(badOverallList), DATA_AMOUNT))

            elif data[i]['overall'] == 5.0 and len(goodOverallList) < DATA_AMOUNT:
                goodOverallList.append(data[i]['overall'])
                goodReviewTextList.append(data[i]['reviewText'])
                print('Data {}/{} Loaded!(5.0)'.format(len(goodOverallList), DATA_AMOUNT))
            
            else:
                if len(badOverallList) >= DATA_AMOUNT and len(badReviewTextList) >= DATA_AMOUNT:
                    break
                else:
                    print('Data {} is unused.'.format(i))         

    overallList = goodOverallList + badOverallList
    reviewTextList = goodReviewTextList + badReviewTextList

    for _ in range(DATA_AMOUNT * 2):
        TEMP.append([overallList[_], reviewTextList[_]])
    
    random.shuffle(TEMP)

    for _ in range(DATA_AMOUNT * 2):
        overallList[_] = TEMP[_][0]
        reviewTextList[_] = TEMP[_][1]

    reviewTextList = NLP(reviewTextList)
    encodedWords = IntegerEncode(reviewTextList)

    for j in range(len(overallList)):
        temp = []

        ('Data {}/{} Appended!'.format(j, len(overallList)))
        preprocessed['overall'].append(overallList[j])
        
        for k in range(len(reviewTextList[j])):
            temp.append(encodedWords[reviewTextList[j][k]])

        preprocessed['reviewText'].append(temp)

    with open('data.json', 'w') as f:
        json.dump(preprocessed, f, indent = 4)

    with open('Encode.json', 'w') as f:
        json.dump(encodedWords, f, indent = 4)
