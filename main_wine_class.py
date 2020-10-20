###################################
#ACKNOWLEDGEMENTS
###################################
"""
Thanks to Youssef for his input and support on the project!
"""
###################################
#IMPORT STATEMENTS
###################################
import pandas as pd
from collections import Counter
from math import isnan
import numpy as np
import matplotlib.pyplot as plt
import pickle
import string
from sklearn.model_selection import train_test_split

#nlp stuff
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

#to save my poor little CPU
import nbconvert
import multiprocessing

#vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

#machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression

#validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#setting the price categories. Below 10, 0, 10-20, 1...
#price_cat = {2000:9, 1000:8, 500:7, 200:6, 100:5, 50:4, 30:3, 20:2, 10:1, 0:0}
price_cat = {1000:7, 300:6, 150:5, 80:4, 40:3, 20:2, 0:1}
###################################
#DATA PREPROCESSING
###################################

#import the dataset and getting the columns we want
def import_data(filename):
    master_set = pd.read_csv('winemag-data-130k-v2.csv')
    X_raw = pd.DataFrame(master_set['description'])
    y_cat = pd.DataFrame(master_set['price'].astype('float16'))
    return (X_raw, y_cat)

#cleans the datasets from null values extra characters. Lowers
def clean_sets(X_raw, y_cat):
    #merge sets to remove the NaN values
    merged_sets = X_raw.merge(y_cat, on=None, how='left', left_index=True, right_index=True)
    merged_sets = merged_sets[merged_sets['price'].notna()]
    X_raw = pd.DataFrame(merged_sets['description'])
    y_cat = pd.DataFrame(merged_sets['price'])
    #remove punctuation
    translation_table = str.maketrans('', '', string.punctuation)
    X_raw['description'] = [review.lower().translate(translation_table) for review in X_raw['description']]
    return (X_raw, y_cat)

#tokenize and lemmatize the input set
def tokenize_and_lemmatize(X_set):
    #tokenize
    X_set['description'] = [word_tokenize(review) for review in X_set['description']]
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    X_set['description'] = [[word for word in review if not word in stop_words] for review in X_set['description']]    
    #lemmatize each review
    X_set['description'] = [lemmatize_review(review) for review in X_set['description']]
    return X_set

#helper function for tokenize_and_lemmatize. Lemmatizes a review
def lemmatize_review(review):
    review = pos_tag(review)
    lemmatizer = WordNetLemmatizer()
    lemmatized_review = [lemmatizer.lemmatize(word, pos=correct_pos_tag(tag)) for word, tag in review]
    return lemmatized_review
    
#helper function for lemmatize_review. Maps all pos tags to basic categories
def correct_pos_tag(tag):
    tag = tag[0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N':wordnet.NOUN, 'V':wordnet.VERB, 'R':wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) #returns a noun by default

#helper function to categorize a price in a category
def to_price_cat(bottle_price):
    for tresh_price in price_cat.keys():
        if bottle_price >= tresh_price:
            return price_cat[tresh_price]

#function to categorize the prices in a dataset
def categorize_prices(prices):
    prices  = [to_price_cat(price) for price in prices]
    return prices

###################################
#SAVE AND LOAD FUNCTIONS
###################################

#function to save/retrieve the data for easy retrieval afterwards
def save_data(X, y, filename):
    with open(filename, 'wb') as f:
        pickle.dump( [X, y], f)
    print("Data saved to {}".format(filename))
    
    
def load_data(filename):
    with open(filename, 'rb') as f:
        [X_raw, y_cat] = pickle.load(f)
    return (X_raw, y_cat)

###################################
#VECTORIZER FUNCTION
###################################

#creates a word2vec embedding model
def create_word_embedding():
    #count number of cores
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(window=2, size=300, alpha=0.03, workers=cores-1)
    return w2v_model

#TF_IDF vectorization
def dummy(sentence):
    return sentence
def tfidf_vectorize():
    vectorizer = TfidfVectorizer(tokenizer=dummy, analyzer='word', preprocessor=dummy, token_pattern=None,max_features=5000)
    vectorizer.fit(X_raw['description'])
    X_train_Tfidf = vectorizer.transform(X_train['description'])
    X_test_Tfidf = vectorizer.transform(X_test['description'])
    return (X_train_Tfidf, X_test_Tfidf)

###################################
#MACHINE LEARNING MODELS
###################################
def naive_bayes_classifier(X_set, y_set):
    naive = naive_bayes.MultinomialNB()
    naive.fit(X_set, y_set)
    return naive
def rfc_classifier(X_set, y_set):
    cores = multiprocessing.cpu_count()
    rfc = RandomForestClassifier(n_jobs=cores-1)
    rfc.fit(X_set, y_set)
    return rfc
def logistic_regression_classifier(X_set, y_set):
    cores = multiprocessing.cpu_count()
    lgc = LogisticRegression(multi_class='multinomial', n_jobs=cores-1)
    lgc.fit(X_set, y_set)
    return lgc
###################################
#PROGRAM EXECUTION
###################################

#the main program
#(X_raw, y_cat) = import_data('winemag-data-130k-v2.csv')

#cleaner functions
#X_raw, y_cat = clean_sets(X_raw, y_cat)
#y_cat['price'] = categorize_prices(y_cat['price'])
#X_raw = tokenize_and_lemmatize(X_raw)

#load data/save data
#save_data(X_raw, y_cat, 'data_lemmatized.txt')
X_raw, y_cat = load_data('data_lemmatized.txt')
#split data
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_cat, test_size = 0.1)
(X_train_Tfidf, X_test_Tfidf) = tfidf_vectorize()
#apply naive bayes

naive_model = naive_bayes_classifier(X_train_Tfidf, y_train['price'])
naive_predicted = naive_model.predict(X_test_Tfidf)
train_pred = naive_model.predict(X_train_Tfidf)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, naive_predicted)
print("The test accuracy of the Naive Bayes model is {}%".format(test_accuracy*100))
print("The train set accuracy of the Naive Bayes model is {}%".format(train_accuracy*100))

#apply random forest classifier
rfc = rfc_classifier(X_train_Tfidf, y_train['price'])
rfc_predicted = rfc.predict(X_test_Tfidf)
train_predict = rfc.predict(X_train_Tfidf)
train_accuracy = accuracy_score(y_train, train_predict)
test_accuracy = accuracy_score(y_test, rfc_predicted)
print("The test accuracy of the RFC model is {}%".format(test_accuracy*100))
print("The train set accuracy of the RFC model is {}%".format(train_accuracy*100))

#apply logistic regression

lgc = logistic_regression_classifier(X_train_Tfidf, y_train['price'])
lgc_predicted = lgc.predict(X_test_Tfidf)
train_pred = lgc.predict(X_train_Tfidf)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, lgc_predicted)
print("The test accuracy of the Logistic Regression model is {}%".format(test_accuracy*100))
print("The train set accuracy of the Logistic Regression model is {}%".format(train_accuracy*100))


#Confusion matrices
C_naive = confusion_matrix(y_test, naive_predicted)
C_rfc = confusion_matrix(y_test, rfc_predicted)
C_lgc = confusion_matrix(y_test, lgc_predicted)
print("Naive Bayes confusion matrix:")
print(C_naive)
print("RFC confusion matrix:")
print(C_rfc)
print("LGC confusion matrix:")
print(C_lgc)
f, a= plt.subplots(1,3)
a[0].imshow(C_naive, cmap='binary')
a[0].set_title("Naive Bayes CM")
a[1].imshow(C_rfc, cmap='binary')
a[1].set_title("RFC CM")
a[2].imshow(C_lgc, cmap='binary')
a[2].set_title("LGC CM")
plt.show()
