'''
Natural Language Processing Project
Yelp Reviews data set from Kaggle
'''
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

def gather_data(file):
    yelp = pd.read_csv(file)
    return yelp


def process_data(yelp, display=False):
    if display:
        print(yelp.head(10))
        print(yelp.info())
        print(yelp.describe())

    yelp['text length'] = yelp['text'].apply(len)
    return yelp


def explore_data(yelp):
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

def nlp_classifier(yelp):
    yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]
    X = yelp_class['text']
    y = yelp_class['stars']
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    return X, y

def training_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 101)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb


def evaluation(nb, X_test, y_test):
    preds = nb.predict(X_test)
    print(confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))

if __name__ == '__main__':
    filename = 'yelp.csv'
    yelp = gather_data(filename)
    processed_yelp = process_data(yelp)
    #explore_data(processed_yelp)
    X_and_y = nlp_classifier(processed_yelp)
    X = X_and_y[0]
    y = X_and_y[1]
    splits = training_split(X, y)
    X_train = splits[0]
    X_test = splits[1]
    y_train = splits[2]
    y_test = splits[3]
    nb = train_model(X_train, y_train)
    evaluation(nb, X_test, y_test)