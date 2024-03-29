# Data preprocessing

import numpy as np
import pandas as pd

dataset = pd.read_csv('dow_jones_index.data')

print(dataset)

dataset[dataset.columns[3:]] = dataset[dataset.columns[3:]].replace('[\$]', '', regex=True).astype(float)

print("updated dataset")
print(dataset)

## Filling up the missing entries
from sklearn.preprocessing import Imputer
# mean, median or most_frequent
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(dataset[dataset.columns[3:]])
dataset[dataset.columns[3:]]= imputer.transform(dataset[dataset.columns[3:]])

print("full dataset")
print(dataset)

# so, pre_dataset's NaN entries has been computed. Now we will categorized each companies' name

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset[dataset.columns[1]] = labelencoder.fit_transform(dataset[dataset.columns[1]])

print("categorized dataset")
print(dataset)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
dataset[dataset.columns[3:]] = sc_X.fit_transform(dataset[dataset.columns[3:]])

print("feature scaled dataset")
print(dataset)

# Uncomment this if you do not want date in the dataset
#dataset = dataset.drop('date', 1)
#print("Without date")
#print(dataset)

# Splitting the data dataset
from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 0.2, random_state = 0)
