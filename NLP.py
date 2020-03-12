from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer as st
from nltk.stem import WordNetLemmatizer as lm
from nltk.tag import pos_tag
from nltk.corpus import stopwords as stw

def Pretreatment(data):
    newData = []

    for sentence in data:
        newSentence = []

        sentence = RegexpTokenizer('[\w]+').tokenize(data)

        sentence = [st().stem(w) for w in sentence]
        sentence = [lm().lemmatize(w) for w in sentence]

        for w in sentence:
            if w.lower() not in stw.words('english'):
                newSentence.append(w)

        newData.append(newSentence)
    
    return newData

if __name__ == "__main__":
    data = 'In grammar, inflection is the modification of a word to express different grammatical categories such as tense,' + \
           'case, voice, aspect, person, number, gender, and mood. An inflection expresses one or more grammatical categories with a prefix,' + \
           'suffix or infix, or another internal modification such as a vowel change'

    print(Pretreatment(data))