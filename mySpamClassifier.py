# Naive Bayes Classifer
# Create classifier with methods train, classify, and accuracy.
# Retrieve and label spam and ham documents from spam and ham directories.
# The features are the 2000 most common words in all the documents.
# Each feature gets a say in deciding which label should be assigned to a given input value.
# Starts by calculating prior probability of each label, determined by the frequency of
# each label in the training set, e.g. 60 spam and 40 ham out of 100 files, 
# spam has a 60% prior probability and ham has a 40% prior probability.
# Each feature contributes to the prior probability to get a likelihood estimate foe each label.
# The label with the highest likelihood estimate is assigned to the input value, 
# e.g. 39% estimate for spam, 61% estimate for ham, file is assigned "ham".
import os
import random
import time

import nltk
from nltk.corpus import stopwords

stopwords = nltk.corpus.stopwords.words('english')  # all non-descriptive English words


class mySpamClassifier:
    def __init__(self, spamFolder, hamFolder):
        self.totalSpamWords = []
        self.totalHamWords = []
        self.totalSpamFile = []
        self.spamFiles = os.listdir(spamFolder)
        self.totalHamFile = []
        self.hamFiles = os.listdir(hamFolder)
        self.spamWord = {}
        self.hamWord = {}
        self.classifiedList = []
        self.spamChance = 0
        self.hamChance = 0

        i = 0
        print("Reading in spam")
        lastPerc = 0
        for docs in self.spamFiles:
            percDone = (i / float(len(self.spamFiles))) * 100
            if percDone % 20 <= .1 and int(percDone) != lastPerc:
                lastPerc = int(percDone)
                print("%s%% done" % int(percDone))
            i += 1
            textFile = open(spamFolder + "/" + docs, "r")
            lines = textFile.readlines()
            textFile.close()
            wordList = [w.split() for w in lines]  # splits the lines into words
            words = sum(wordList, [])  # flattens inot a simple list of all words
            featureWords = [w.lower() for w in words if
                            w not in stopwords and len(w) > 1 and w.isalpha()]  # omits needless words
            featureWords = list(set(featureWords))  # remove duplicates
            self.totalSpamFile.append((featureWords, "spam"))  # assigns label for file
            self.totalSpamWords += featureWords  # adds words to total words

        i = 0
        lastPerc = 0
        print("Reading in ham")
        for docs in self.hamFiles:
            percDone = (i / float(len(self.hamFiles))) * 100
            if percDone % 20 <= .1 and int(percDone) != lastPerc:
                lastPerc = int(percDone)
                print("%s%% done" % int(percDone))
            i += 1
            textFile = open(hamFolder + "/" + docs, "r")
            lines = textFile.readlines()
            textFile.close()
            wordList = [w.split() for w in lines]  # splits the lines into words
            words = sum(wordList, [])  # flattens inot a simple list of all words
            featureWords = [w.lower() for w in words if
                            w not in stopwords and len(w) > 1 and w.isalpha()]  # omits needless words
            featureWords = list(set(featureWords))  # remove duplicates
            self.totalHamFile.append((featureWords, "ham"))  # assigns label for file
            self.totalHamWords += featureWords  # adds words to total words

        self.documents = self.totalSpamFile
        self.documents += self.totalHamFile
        random.shuffle(self.documents)  # list with spam and ham documents randomly distributed

        certainIndex = int(len(self.documents) * 0.9)  # getting 90% and 10% of the total documents
        self.trainDocs = self.documents[:certainIndex]  # 90% of total documents
        self.testDocs = self.documents[certainIndex:]  # 10% of total documents
        self.totalWords = self.totalSpamWords + self.totalHamWords
        random.shuffle(self.totalWords)
        self.all_words = nltk.FreqDist(w for w in self.totalWords if w.isalpha())  # lists frequency of all words
        self.word_features = list(self.all_words)[:2000]  # lists top 2000 most frequent words

    def train(self):  # trains the classifier by calculating probabilities
        print("Starting to train")
        numSpam = 0
        for doc in self.trainDocs:
            if doc[1] == "spam":
                numSpam += 1
        self.spamChance = numSpam / len(self.trainDocs)
        self.hamChance = 1 - self.spamChance

        for i in range(len(self.word_features)):
            percDone = (i / float(len(self.word_features))) * 100
            if percDone % 10 <= .01:
                print("%s%% done" % percDone)
            word = self.word_features[i]
            self.spamWord.setdefault(word, 0)
            self.hamWord.setdefault(word, 0)
            hnum = 0
            snum = 0
            for doc in self.trainDocs:
                num = doc[0].count(word)
                if doc[1] == "spam":
                    snum += num
                else:
                    hnum += num

            hPerc = hnum / (len(self.totalHamWords))
            sPerc = snum / len(self.totalSpamWords)
            hPerc = .1 if hPerc == 0 else hPerc
            sPerc = .1 if sPerc == 0 else sPerc
            self.spamWord[word] = hPerc
            self.hamWord[word] = sPerc

    def classify(self):  # labels test docs as spam or ham based on feature probs.
        for doc in self.trainDocs:
            spamChance = 0
            for word in self.word_features:
                if word in doc[0]:
                    # TODO: Fix this ...
                    spamChance += (self.spamWord[word] * self.spamChance) / (
                        (self.spamWord[word] * self.spamChance) + (self.hamWord[word] * self.hamChance))
            spamChance /= len(self.word_features)
            if spamChance >= .5:
                self.classifiedList.append((doc[0], "spam"))
            else:
                self.classifiedList.append((doc[0], "ham"))
        return self.classifiedList

    def accuracy(self):  # calculates percent of docs that were correctly classified
        result = 0
        for i in range(len(self.trainDocs)):
            doc = self.trainDocs[i]
            if self.classifiedList[i][1] == doc[1]:
                result += 1
        result /= float(len(self.trainDocs))
        return result


if __name__ == '__main__':
    c = mySpamClassifier("spam", "ham")
    start = time.time()
    c.train()
    end = time.time()
    c.classify()
    print("Accuracy is %s%%" % (c.accuracy() * 100))
    print("Took %s seconds to train" % int(end - start))
