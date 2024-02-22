#from sb_corpus_reader import SBCorpusReader
#import nltk
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
failed_words = []

def convert_to_dict(words):
    dict = {}
    for i in range(0, len(words), 2):
        dict[words[i][0]] = words[i][1]
    return dict


def classify_data(text, lib, no_NA):
    # Input training data and dict of what word class a word is
    # Returns a dict of the input data with matched word classes.
    classlist = []
    sentences = text.lower().split('. ')  #FIX: Last word in each sentence has a "."

    NA_list = []
    for sentence in sentences:
        words = sentence.split(' ')
        for word in words:
            try:
                classlist.append(lib[word])
            except:
                print(word)
                classlist.append('NA')
                NA_list.append('NA')
        #print(len(NA_list))
        if no_NA:
            if len(NA_list)>0:
                #print(classlist)
                #print(len(words))
                for _ in range(len(words)): # Dont understand this!
                    classlist.pop(-1)
                NA_list = []
            else:
                classlist.append('.')
        else:
            classlist.append('.')
    return classlist




def save_dict(to_save):
    json_object = json.dumps(to_save, indent=4)
    with open("dictionaries/classdict.json", "w") as outfile:
        outfile.write(json_object)


def open_dict(file):
    with open(file, 'r') as openfile:
        # Reading from json file
        return json.load(openfile)


def read_traning_csv(file):
    abstracts = []
    df = pd.read_csv(file, usecols=['Abstract'])
    for i in range(len(df)):
        if str(df.iloc[i][0]) != "nan":
            abstracts.append(str(df.iloc[i][0]))
    return abstracts

def read_translation_txt(file):
    with open(file, 'r', encoding='utf-8') as file:
        # Reading a text file
        return file.read()

def scrape_and_save():
    abstracts = []
    raw = open_dict('abstractsmalltraining.json')
    for paper in raw:
        abstracts.append(paper['abstract'])
    return abstracts


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


def check_english(text):
    """Checks if the text contain any of these common english words (which don't occur in swedish)"""
    common_english_words = ["the", "be", "to", "of", "and", "a", "that"]
    english = False
    for word in text:
        if word in common_english_words:
            english = True
            break
    return english


def text_cleaner(text, no_dot):
    #In the future we will want to preserve symbols which are supposed to be there such as comas
    """Cleans text from symbols hindering word class identification"""
    text = text.replace('- ', '')
    text = text.lower()
    text = text.split()
    first_clean = []
    common_long_clutter = ['<p>', '.</p>']
    for words in text:
        for substring in common_long_clutter:
            words = words.replace(substring, "")
        first_clean.append(words)
    # Then removes all unwanted characters
    if no_dot:
        common_clutter = "a b c d e f g h i j k l m n o p q r s t u v w x y z å ä ö".split() # Removed the dot, i think it sitll works
    else:
        common_clutter = "a b c d e f g h i j k l m n o p q r s t u v w x y z å ä ö .".split() # Removed the dot, i think it sitll works

    new_text_list = []
    for word in first_clean:
        new_word = ""
        for letter in word:
            if letter in common_clutter:
                new_word += letter
        new_text_list.append(new_word)
    new_text = ' '.join(new_text_list)
    return new_text


def test():
    dictionary_talbanken = open_dict('dictionaries/classdict.json')
    text = read_traning_csv('export1.csv')[0]
    classified = classify_data(text, dictionary_talbanken)
    k = 0
    for i in classified:
        if i == 'NA':
            k += 1
    #print("Number of words that could not me classified: " + str(k) + " out of " + str(len(classified)))

def abstracts_to_word_classes(file, no_NA):
    """Converts the text to word classes"""
    k = 0  # Counting amount of unclassified words
    classified_list = []  # [text, text, text..] (with text in word class form)
    word_class_list = []  # [all texts] (with text in word class form)
    dictionary_talbanken = open_dict('dictionaries/classdict.json')
    text_all = read_traning_csv(file)
    for text in text_all:
        if check_english(text.split()):
            continue
        text = text_cleaner(text)
        classified = classify_data(text, dictionary_talbanken, no_NA)
        for i in classified:
            if i == 'NA':
                k += 1
        classified_list.append(classified)
    for sublist in classified_list:
        for word_class in sublist:
            word_class_list.append(word_class)
    print("Number of words that could not be classified: " + str(k) + " out of " + str(len(word_class_list)))
    print("in percent " + str(100*k/len(word_class_list)))
    with open('wordclasslists/WC_all.json', "w") as outfile:
        json.dump(word_class_list, outfile)
    #print(word_class_list)
    return word_class_list


def translations_to_word_classes(file, filename, no_NA):
    # Takes a txt file and converts it to word classes
    dictionary_talbanken = open_dict('dictionaries/classdict.json')
    text = read_translation_txt(file)
    text = text_cleaner(text)
    classified = classify_data(text, dictionary_talbanken, no_NA)
    k = 0
    for i in classified:
        if i == 'NA':
            k += 1
    print("Number of words that could not be classified: " + str(k) + " out of " + str(len(classified)))
    print("in percent " + str(100 * k / len(classified)))
    with open(filename, "w") as outfile:
        json.dump(classified, outfile)
    return classified


def unique_word_classes():
    # returns set of all unique word classes
    large_list = []
    unique_codes = set()
    dictionary_talbanken = open_dict('clas.split()sdict.json')
    text = read_traning_csv('Trainingdata/many_abstracts.csv')
    for elem in text:
        large_list.append(classify_data(elem, dictionary_talbanken))
    for sublist in large_list:
        for word_class in sublist:
            unique_codes.add(word_class)

    return unique_codes


if __name__ == "__main__":
    """For joining new files to the large word to word-class dictionary"""
    # fl = joindicts()
    # save_dict(fl)
    
    """Translates web-scraped csv files to word classes"""
    #abstracts_to_word_classes('Trainingdata/many_abstracts.csv', False)

    """Translates txt file to word classes"""
    #translations_to_word_classes('Trainingdata/real_sample.txt', 'wordclasslists/WC_non_transl.json')
    translations_to_word_classes('Trainingdata/translated_sample.txt', "wordclasslists/WC_transl.json", False)

    """
    TM_all = open_dict('transition_matrices/TM_all')
    maxlike(TM_all, sample)"""

