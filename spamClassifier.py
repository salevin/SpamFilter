import nltk
import os
import random
from nltk.corpus import stopwords

stopwords = nltk.corpus.stopwords.words('english')  # irrelevant little english words
totalSpamWords = []
totalSpamFile = []
spamFiles = os.listdir("spam")

# process each spam file, collecting spam feature words
for docs in spamFiles:
    textFile = open("spam/" + docs, "r")
    lines = textFile.readlines()
    textFile.close()
    wordList = [w.split() for w in lines]  # split the lines into words
    words = sum(wordList, [])  # flatten into a simple list of all words
    featureWords = [w.lower() for w in words if
                    w not in stopwords and len(w) > 1 and w.isalpha()]  # omit needless words
    totalSpamFile.append((featureWords, "spam"))
    totalSpamWords += featureWords

totalHamWords = []
totalHamFile = []
hamFiles = os.listdir("ham")

# process each ham file, collecting ham feature words
for docs in hamFiles:
    textFile = open("ham/" + docs, "r")
    lines = textFile.readlines()
    textFile.close()
    wordList = [w.split() for w in lines]
    words = sum(wordList, [])
    featureWords = [w.lower() for w in words if w not in stopwords and len(w) > 1]
    totalHamFile.append((featureWords, "ham"))
    totalHamWords += featureWords

documents = totalSpamFile
documents += totalHamFile
random.shuffle(documents)

# get the top 2000 most frequent words in all documents	
totalWords = totalSpamWords + totalHamWords
random.shuffle(totalWords)
allWords = nltk.FreqDist(w for w in totalWords if w.isalpha())
word_features = list(allWords)[:2000]


# computes the feature set (the words that are in most common) from a given document
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


featuresets = [(document_features(d), c) for (d, c) in documents]

# split the documents into training and test sets
train_set, test_set = featuresets[4000:], featuresets[:4000]

# create a classifier using NaiveBayes with the training set
classifier = nltk.NaiveBayesClassifier.train(train_set)

# classify the test set and check the accuracy
print(nltk.classify.accuracy(classifier, test_set))

# which features are most informative
classifier.show_most_informative_features(20)
