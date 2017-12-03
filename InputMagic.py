# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import Doc2Vec as d2v

# numpy, etc
import numpy
import json

# random
from random import shuffle as sf

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
        sf(self.sentences)
        return self.sentences

# Doc2Vec model
def generate_doc2vec_model(sources, name, lang_tag):
    doc = LabeledLineSentence(sources)
    for line in doc:
        print(line) #Debug
    # size: 100
    # min_alpha = 0.025
    model = d2v( size=300, window=10, min_count=1, sample=1e-4, negative=5,workers=8)
    model.build_vocab(doc.to_array())

    alpha_val = 0.025        # Initial learning rate
    min_alpha_val = 1e-4     # Minimum for linear learning rate decay
    passes = 15              # Number of passes of one document during training

    alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)

    for epoch in range(passes):
        model.alpha, model.min_alpha = alpha_val, alpha_val
        model.train(doc.shuffle(),total_examples=model.corpus_count, epochs=model.iter)
        # Logs
        print('Completed pass %i at alpha %f' % (epoch + 1, alpha_val))
        # Next run alpha
        alpha_val -= alpha_delta
    model.save(name+'-model-'+lang_tag+'.d2v')

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

## Section 1: Preprocessing Articles
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
            # cleaning up additional space
            if cleanRaw[0] == " ":
                cleanRaw = cleanRaw[1:len(cleanRaw)]

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

## Section 2: Doc2Vec training model
os.chdir('..')

modals_path = "trained_modals"

# Adding gitignore to this path
f = open(".gitignore", "a")
f.write("%s/" % modals_path)
f.write("\n")
f.close()

print(os.getcwd()+'/cleaned_articles')
top = os.getcwd()+'/cleaned_articles'

try:
    os.makedirs(modals_path)
except OSError:
    # Prevent race condition
    if not os.path.isdir(modals_path):
        raise
os.chdir(modals_path)

## get articles and put into doc2vec

for root, dirs, files in os.walk(top):
    for d in dirs:
        print("Now we are at: ")
        print(d)
        # Iterating companies in /cleaned_articles/
        sources = {}
        count = 1
        cwd = top + "/%s" % d
        for root, dirs, files in os.walk(cwd):
            for filename in files:
                print("File: %s"%filename)
                article_name_tag = d + "_%d" % count
                article_file_location = "../{}/{}/{}".format(total_articles_path,d,filename)
                sources[article_file_location] = article_name_tag
                count += 1
        # Then, get the files and put into sources
        generate_doc2vec_model(sources, d, "en")
    break


# To load:
# model_loaded = d2v.load('/tmp/my_model.d2v')
