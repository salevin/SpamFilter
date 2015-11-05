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

#        \`*-.
#         )  _`-.
#        .  : `. .
#        : _   '  \
#        ; *` _.   `*-._
#        `-.-'          `-.
#          ;       `       `.
#          :.       .        \
#          . \  .   :   .-'   .
#          '  `+.;  ;  '      :
#          :  '  |    ;       ;-.
#          ; '   : :`-:     _.`* ;
# [bug] .*' /  .*' ; .*`- +'  `*'
#       `*-*   `*-*  `*-*'

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
        # Get probability of any random document being spam
        for doc in self.trainDocs:
            if doc[1] == "spam":
                numSpam += 1
        self.spamChance = float(numSpam) / len(self.trainDocs)
        # Ham chance is just the inverse of the spam chance
        self.hamChance = 1 - self.spamChance

        minHam = (1.0 / (len(self.trainDocs) - numSpam + 1))
        minSpam = (1.0 / (numSpam + 1))

        # Start the real training
        for i in range(len(self.word_features)):

            # Read out how far along training is
            percDone = (i / float(len(self.word_features))) * 100
            if percDone % 10 <= .01:
                print("%s%% done" % percDone)

            # Look at each word in the featured words
            word = self.word_features[i]

            self.spamWord.setdefault(word, minSpam)
            self.hamWord.setdefault(word, minHam)

            # Initialize number of spam and ham occurrences to zero
            hnum = 0
            snum = 0

            # Add up occurrences
            for doc in self.trainDocs:
                if word in doc[0]:
                    if doc[1] == "spam":
                        snum += 1
                    else:
                        hnum += 1

            # Get the percentage value
            hPerc = float(hnum) / (len(self.trainDocs) - numSpam)
            sPerc = float(snum) / numSpam

            # Make sure we aren't setting them to zero
            if hPerc != 0:
                self.hamWord[word] = hPerc
            if sPerc != 0:
                self.spamWord[word] = sPerc

    def classify(self):  # labels test docs as spam or ham based on feature probs.
        for doc in self.testDocs:
            hamProb = self.hamChance
            spamProb = self.spamChance
            for word in doc[0]:
                if word in self.word_features:
                    hamProb *= self.hamWord[word]
                    spamProb *= self.spamWord[word]

            if spamProb > hamProb:
                self.classifiedList.append((doc[0], "spam"))
            else:
                self.classifiedList.append((doc[0], "ham"))

    def accuracy(self):  # calculates percent of docs that were correctly classified
        result = 0
        for i in range(len(self.testDocs)):
            doc = self.testDocs[i]
            if self.classifiedList[i][1] == doc[1]:
                result += 1
        result /= float(len(self.testDocs))
        return result


if __name__ == '__main__':
    c = mySpamClassifier("spam", "ham")
    start = time.time()
    c.train()
    end = time.time()
    print("Done training")
    c.classify()
    accuracy = c.accuracy() * 100
    print("Accuracy is %s%%" % accuracy)
    if accuracy > 80:
        print("doot doot you win")
    print("Took %s seconds to train" % int(end - start))
