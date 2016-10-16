import re


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


def train_snake_language(cl):
    cl.train('Pythons are constrictors that feed on birds and mammals', 'Snake')
    cl.train('Python was originally developed as a scripting language', 'Language')
    cl.train('A 49-ft.-long python was found in Indonesia', 'Snake')
    cl.train('Python has a dynamic type system', 'Language')
    cl.train('Python with vivid scales', 'Snake')
    cl.train('Open source project', 'Language')


def print_classify(cl, text):
    class1 = "Language"
    class2 = "Snake"
    print text + " -> " + cl.classify(text, default="unknown")
    print class1 + " with probability " + str(cl.prob(text, class1))
    print class2 + " with probability " + str(cl.prob(text, class2))
    print


if __name__ == '__main__':
    cl = naivebayes(getwords)
    train_snake_language(cl)
    print_classify(cl, "dynamic programming")
    print_classify(cl, "boa constrictors")
