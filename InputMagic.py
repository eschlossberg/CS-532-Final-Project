# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy, etc
import numpy
import json

## For Debug purpose
from pprint import pprint

## Cleaning the datas
data = json.load(open('companies.json'))

pprint(data)
