import nltk
from sb_corpus_reader import SBCorpusReader
talbanken = SBCorpusReader('Classified_word_lists/talbanken.xml')
talbanken.sents()
talbanken_words = (talbanken.tagged_words())

def classify_data():
    # Input training data and dict of what word class a word is
    # Returns a dict of the input data with matched word classes.
    pass
# Here have help functions for this one, this should be the main one that returns what we want


def word_class_dict():
    # Reutus dict of wordclasses
    pass


import nltk
import ssl
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
    sentences = text.lower().split('. ')
    for sentence in sentences:
        words = sentence.split(' ')
        for word in words:
            print(word)
            classlist.append(lib[word])
        classlist.append('.')

    return classlist

dictionary_talbanken = convert_to_dict(talbanken_words)
text = "du en. och"
print(classify_data(text, dictionary_talbanken))