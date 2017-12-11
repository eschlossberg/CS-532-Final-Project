# numpy
import numpy

# random
from random import shuffle as sf

# For Debug purpose
from pprint import pprint

# For cleaning texts
import re

# For directory
import os

print(os.getcwd()+'/Data')
top = os.getcwd()+'/Data'
etf = top + '/ETFs'
stocks = top + '/Stocks'

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
                sources[filename] = article_name_tag
                count += 1

        print("Collected sources: ")
        print(sources)
    break
