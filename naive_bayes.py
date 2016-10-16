import re

import sys
from os.path import splitext


def getwords(doc):
    splitter = re.compile('\\W*')
    # Split the words by non-alpha characters
    words = [s.lower() for s in splitter.split(doc) if 2 < len(s) < 20]

    # Return the unique set of words only
    return dict([(w, 1) for w in words])


class classifier:
    def __init__(self, getfeatures, filename=None):
        # Counts of feature/category combinations
        self.fc = {}
        # Counts of documents in each category
        self.cc = {}
        self.getfeatures = getfeatures

    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    def totalcount(self):
        return sum(self.cc.values())

    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        features = self.getfeatures(item)
        # Increment the count for every feature with this category
        for f in features:
            self.incf(f, cat)
        # Increment the count for this category
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0
        # The total number of times this feature appeared in this
        # category divided by the total number of items in this category
        return self.fcount(f, cat) / self.catcount(cat)

    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        # Calculate current probability
        basicprob = prf(f, cat)
        # Count the number of times this feature has appeared in
        # all categories
        totals = sum([self.fcount(f, c) for c in self.categories()])
        # Calculate the weighted average
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp


class naivebayes(classifier):
    def __init__(self, getfeatures):
        classifier.__init__(self, getfeatures)
        self.thresholds = {}

    def docprob(self, item, cat):
        features = self.getfeatures(item)
        # Multiply the probabilities of all the features together
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        catprob = self.catcount(cat) / self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob * catprob

    def setthreshold(self, cat, t):
        self.thresholds[cat] = t

    def getthreshold(self, cat):
        if cat not in self.thresholds:
            return 1.0
        return self.thresholds[cat]

    def classify(self, item, default=None):
        probs = {}
        # Find the category with the highest probability
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat
        # Make sure the probability exceeds threshold*next best
        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.getthreshold(best) > probs[best]:
                return default
        return best


def train(cl, trainDict):
    for classLabel, sentencesList in trainDict.iteritems():
        for sentence in sentencesList:
            cl.train(sentence, classLabel)


def get_classification_response(cl, text):
    categories = cl.categories()
    response = text + " -> " + cl.classify(text, default="unknown") + "\n"
    for category in categories:
        response += category + " with probability " + str(cl.prob(text, category)) + "\n"
    return response


def get_input_file_name():
    if len(sys.argv) != 2:
        print "Need one dataset file name:\n" \
              "Example:\n" \
              "python naive_bayes.py file.txt"
        exit(0)
    return sys.argv[1]


def parse_input(file):
    categorySentenceDict = {}
    for line in file:
        splitLine = line.split()
        category = splitLine[0]
        sentence = " ".join(splitLine)
        if category in categorySentenceDict:
            categorySentenceDict[category].append(sentence)
        else:
            categorySentenceDict[category] = [sentence]
    return categorySentenceDict


def get_dict_from_input_file():
    inputFileName = get_input_file_name()
    with open(inputFileName, 'r') as datasetFile:
        trainDict = parse_input(datasetFile)
    logFileName = splitext(inputFileName)[0] + ".log"
    return trainDict, logFileName


if __name__ == '__main__':
    trainDict, logFileName = get_dict_from_input_file()
    cl = naivebayes(getwords)
    train(cl, trainDict)

    with open(logFileName, "w") as logFile:
        print "For exit press Enter."
        query = raw_input("Input query for classification:\n")
        while query != "":
            response = get_classification_response(cl, query)
            print response
            logFile.write(response)
            query = raw_input("Input query for classification:\n")
