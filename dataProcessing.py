from sb_corpus_reader import SBCorpusReader
import nltk
import json
import requests 
import pandas as pd
 



  



"""
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
"""
def convert_to_dict(words):
    dict = {}
    for i in range(0, len(words), 2):
        dict[words[i][0]] = words[i][1]
    return dict



def classify_data(text, lib):
    # Input training data and dict of what word class a word is
    # Returns a dict of the input data with matched word classes.
    classlist = []
    # Ta bort <p> här också...
    sentences = text.lower().split('. ')
    for sentence in sentences:
        sentence = sentence[:-1]
        words = sentence.split(' ')
        for word in words:
            try:
                classlist.append(lib[word])
            except:
                classlist.append('NA')
        classlist.append('.')

    return classlist


def save_dict(to_save):
    json_object = json.dumps(to_save, indent=4)
    with open("classdict.json", "w") as outfile:
        outfile.write(json_object)

def open_dict(file):
    with open(file, 'r') as openfile:
        # Reading from json file
        return json.load(openfile)
    
def read_traning_csv():
    abstracts = []
    df = pd.read_csv("export.csv", usecols=['Abstract'])
    for i in range(len(df)):
        if str(df.iloc[i][0]) != "nan":
            abstracts.append(str(df.iloc[i][0]))
    return abstracts


def scrape_and_save():
    abstracts = []
    raw = open_dict('abstractsmalltraining.json')
    for paper in raw:
        abstracts.append(paper['abstract'])
    return abstracts


def test():
    dictionary_talbanken = open_dict('classdict.json')
    text = read_traning_csv()[0]
    classified = classify_data(text, dictionary_talbanken)
    k = 0
    for i in classified:
        if i == 'NA':
            k += 1
    print(text)
    print(classified)
    print("Number of words that could not me classified: " + str(k) + " out of " + str(len(classified)))
    # Detta är ganska bra, man behöver ta bort () och liknande, sen är ord som inte identifieras nog nästan alltid substantiv. 


def joindicts():
    talbanken = SBCorpusReader('talbanken.xml')
    talbanken.sents()
    talbanken_words = (talbanken.tagged_words())
    dict1 = convert_to_dict(talbanken_words)
    talbanken = SBCorpusReader('aspacsven-sv.xml')
    talbanken.sents()
    talbanken_words = (talbanken.tagged_words())
    dict2 = convert_to_dict(talbanken_words)
    dict1.update(dict2)
    return dict1

test()
