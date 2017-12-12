# numpy
import numpy as np

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

## All of the stock data stored in the following format:
## {file_name (without .txt) : (data) }

etf_source = {}
stock_source = {}

for root, dirs, files in os.walk(top):
    for d in dirs:
        print("Now we are at: ")
        print(d)
        # Iterating companies in /cleaned_articles/
        sources = {}
        count = 1
        cwd = top + "/%s" % d
        if (d == "Stocks"):
            for root, dirs, files in os.walk(cwd):
                for filename in files:
                    print("We're in stocks")
                    if filename.endswith('.txt'):
                        article_name_tag = os.path.splitext(filename)[0]
                        print(article_name_tag)
                        ## Date, Open, High, Low, Close, Volume, OpenInt
                        fpath = top + "/" + d + "/"+filename
                        if os.path.getsize(fpath) <= 0:
                            continue
                        npdata = np.genfromtxt("./Data/{}/{}".format(d,filename), skip_header=1, delimiter=',', dtype=['U10',float,float,float,float,int,int])
                        #npdata = np.loadtxt("./Data/{}/{}".format(d,filename), skiprows=1, delimiter=',')
                        stock_source[article_name_tag] = npdata

        elif (d == "ETFs"):
                for root, dirs, files in os.walk(cwd):
                    for filename in files:
                        print("We're in ETFs")
                        if filename.endswith('.txt'):
                            article_name_tag = os.path.splitext(filename)[0]
                            print(article_name_tag)
                            ## Date, Open, High, Low, Close, Volume, OpenInt
                            fpath = top + "/" + d + "/"+filename
                            if os.path.getsize(fpath) <= 0:
                                continue
                            npdata = np.genfromtxt("./Data/{}/{}".format(d,filename), skip_header=1, delimiter=',', dtype=['U10',float,float,float,float,int,int])
                            #npdata = np.loadtxt("./Data/{}/{}".format(d,filename), skiprows=1, delimiter=',')
                            etf_source[article_name_tag] = npdata
    break

print(stock_source)
print(etf_source)
