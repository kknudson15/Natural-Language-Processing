'''
Natural Language Processing Project
Yelp Reviews data set from Kaggle
'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk

'''
Process the Data 
'''

yelp = pd.read_csv('yelp.csv')
#print(yelp.head())
#print(yelp.info())
#print(yelp.describe())

yelp['text length'] = yelp['text'].apply(len)
print(yelp.head(10))


'''
Exploratory Data Analysis 
'''

#Facet grid showing the relationship between text length and stars
sns.set_style('dark')
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
plt.show()

#Creates a boxplot of the relationship between stars given and text length
sns.boxplot(x = 'stars', y = 'text length', data = yelp)
plt.show()

#count plot showing the number of each stars given
sns.countplot(x = 'stars',data = yelp)
plt.show()

#groups the data by stars and gets the mean of that data into a new dataframe
stars = yelp.groupby('stars').mean()
print(stars)

#creates a data fram for the correlation data
corr = stars.corr() 
print(corr)

#Creates a heatmap for the correlated data
sns.heatmap(data = corr, cmap = 'coolwarm')
plt.show()

'''
NLP Classification
'''

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

X = yelp_class['text']
y = yelp_class['stars']


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)


'''
Train Test Split
'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 101)

'''
Training a model 
'''
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

'''
Predictions and Evaluations
'''

preds = nb.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))


'''
Using Text Processing 
'''

from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

'''
Using the Pipeline 
'''
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train, y_train)

'''
Predictions and Evaluation 
'''
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

