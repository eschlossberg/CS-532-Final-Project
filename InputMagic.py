# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import d2v

# numpy, etc
import numpy
import json

# For Debug purpose
from pprint import pprint

# For cleaning texts
import re

## Cleaning the datas
data = json.load(open('companies.json'))

print(len(data['companies']))

# Company - CleanArticles pair
company_arr = []
# Clean_articles group format:
#  - Company
#  |-- Article
#  |---Sentences
clean_articles = []

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
    company_arr.append(company['company'])
    clean_articles.append(ca)

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
