# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

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
        test = re.split('[.]', article['text'])
        sentences = []
        for text in test:
            if not text:
                continue
            cleanString = re.sub('\W+', ' ', text)
            if not cleanString:
                continue
            sentences.append(cleanString)
        print(sentences)
        ca.append(sentences)
    company_arr.append(company['company'])
    clean_articles.append(ca)

print(clean_articles)
# Gensim model
