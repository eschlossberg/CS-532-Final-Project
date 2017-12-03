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
clean_articles = []

for company in data['companies']:
    print("Company name: {}  with  {} articles".format(company['company'], len(company['articles'])))
    ca = []
    for article in company['articles']:
        cleanString = re.sub('\W+', ' ', article['text'])
        ca.append(cleanString)

    company_arr.append(company['company'])
    clean_articles.append(ca)

# Gensim model
