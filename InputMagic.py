# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
import gensim.models.doc2vec as d2v

# numpy, etc
import numpy
import json

# For Debug purpose
from pprint import pprint

# For cleaning texts
import re

# For directory
import os

# Copied code, for testing
class LabeledLineSentence(object):
    # sources: a dictionary that has the following format: 'fileName' : 'tag'
    def __init__(self, sources):
        self.sources = sources

        keyChecker = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in keyChecker:
                keyChecker[value] = [key]
            else:
                raise Exception('Non-unique keys detected')

    # Every line has a tag, format: {tag}_{i}
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as doc:
                for i, line in enumerate(doc):
                    yield LabeledSentence( utils.to_unicode(line).split(), [prefix + '_%s' % i])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as doc:
                for i, line in enumerate(doc):
                    self.sentences.append( LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % i]))
        return self.sentences

    def shuffle(self):
        shuffle(self.sentences)
        return self.sentences

# Doc2Vec model
def generate_doc2vec_model(doc, lang_tag):
    doc = LabeledLineSentence(sources)
    for line in doc:
        print(line) #Debug
    # size: 100
    model = d2v(doc, size=300, window=10, min_count=1, sample=1e-4, negative=5,workers=8)
    model.build_vocab(sentences.to_array())
    for epoch in range(10):
        model.train(sentences.shuffle())
    model.save('model-'+lang_tag+'.d2v')

## Execution
# Cleaning the datas
data = json.load(open('companies.json'))

print(len(data['companies']))

# Company - CleanArticles pair
company_arr = []
# Clean_articles group format:
#  - Company
#  |-- Article
#  |---Sentences
clean_articles = []

# make a directory for processed CleanArticles
total_articles_path = "cleaned_articles"

# Adding gitignore to this path
f = open(".gitignore", "w+")
f.write("%s/" % total_articles_path)
f.write("\n")
f.close()

try:
    os.makedirs(total_articles_path)
except OSError:
    # Prevent race condition
    if not os.path.isdir(total_articles_path):
        raise
os.chdir(total_articles_path)



for company in data['companies']:
    print("Company name: {}  with  {} articles".format(company['company'], len(company['articles'])))
    ca = []
    for article in company['articles']:
        raws = re.split('[.]', article['text'])
        sentences = []
        for raw in raws:
            if not raw:
                continue
            cleanRaw = re.sub('\W+', ' ', raw)
            if not cleanRaw:
                continue
            sentences.append(cleanRaw.lower())
        print(sentences)
        ca.append(sentences)

        # Save to file
        # Check path
        # Python 2.7 +
        path = company['company']
        try:
            os.makedirs(path)
        except OSError:
            # Prevent race condition
            if not os.path.isdir(path):
                raise

        f = open("./{}/{}_{}.txt".format(path,path, article['date']), "w+")
        for s in sentences:
            f.write(s)
            f.write("\n")
        f.close()

    # Debug: Make a memory copy of all collections
    company_arr.append(company['company'])
    clean_articles.append(ca)
