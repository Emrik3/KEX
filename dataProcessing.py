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
    # Några ord i lib är upper case
    # Ta bort <p> här också...
    sentences = text.lower().split('. ')
    sentences[0] = sentences[0][3:]
    sentences[-1] = sentences[-1][:-5]
    for sentence in sentences:
        words = sentence.split(' ')
        for word in words:
            try:
                if word[-1] == ',':
                    word = word[:-1]
                classlist.append(lib[word])
            except:
                print(word)
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
    #print(text)
    #print(classified)
    print("Number of words that could not me classified: " + str(k) + " out of " + str(len(classified)))
    # Detta är ganska bra, man behöver ta bort () och liknande, sen är ord som inte identifieras nog nästan alltid substantiv. 


def joindicts():
    # Does not work with some files, like ex. aftonblandet and vetenskap, idk why???
    dict1 = open_dict('classdict.json')
    """talbanken = SBCorpusReader('sweachum.xml')
    talbanken.sents()
    talbanken_words = (talbanken.tagged_words())
    dict2 = convert_to_dict(talbanken_words)"""
    dict2 = open_dict('classdicthum.json')
    dict1.update(dict2)
    return dict1


def test_big_list():
    k = 0 # Counting amount of unclassified words
    classified_list = []  # [text, text, text..] (with text in word class form)
    word_class_list = [] # [all texts] (with text in word class form)
    dictionary_talbanken = open_dict('classdict.json')
    text_all = read_traning_csv()

    for text in text_all:
        classified = classify_data(text, dictionary_talbanken)
        eng =0 # Counter to check if text non-swedish
        for i in range(len(classified)):
            if classified[i] == 'NA':
                k += 1
                if i ==1 or i ==2 or i == 3:
                    eng += 1
                if eng == 3:
                    break
        if eng != 3:
            classified_list.append(classified)
    for sublist in classified_list:
        for word_class in sublist:
            word_class_list.append(word_class)
    #print(word_class_list)
    print("Number of words that could not me classified: " + str(k) + " out of " + str(len(word_class_list)))
    print("in percent " + str(100*k/len(word_class_list)))
    with open("word_classes.json", "w") as outfile:
        json.dump(word_class_list, outfile)




if __name__ == "__main__":
    #fl = joindicts()
    #save_dict(fl)
    #test_big_list()
    test_big_list()
