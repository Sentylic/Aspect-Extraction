# check data existence
import nltk
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

nltk.download('punkt')
from pathlib import Path
from Util import Util
from SenteceToVec import SentenceToVec

stop_words = []

with Path.cwd().joinpath("stop_words_en.txt").open(encoding="utf-8") as f:
    stop_words = f.readlines()
    stop_words = [w.strip() for w in stop_words]

print("{} stop words are read.".format(len(stop_words)))

# Util.convertXMLToJson("Data/ABSA-15_Restaurants_Train.xml", "Data/ABSA-15_Restaurants_Train.json")

label_kinds = []

# make labels (exclude NULL and OOD)
# for e in ["FOOD", "DRINKS", "SERVICE", "AMBIENCE", "LOCATION", "RESTAURANT"]:
#     for a in ["GENERAL", "PRICES", "QUALITY", "STYLE&OPTIONS", "MISCELLANEOUS"]:
#         label_kinds.append(e + "#" + a)
# #         if e in ["market"]:
# #             break;

label_kinds = ["FOOD#QUALITY", "RESTAURANT#GENERAL", "SERVICE#GENERAL", "AMBIENCE#GENERAL",
               "RESTAURANT#MISCELLANEOUS", "FOOD#PRICES", "RESTAURANT#PRICES", "DRINKS#QUALITY",
               "LOCATION#GENERAL", "DRINKS#PRICES"]

print(label_kinds)

import json
import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer

sentences = []
labels = []



with open("Data/ABSA-15_Restaurants_Train.json") as j:
    reviews = json.load(j)['Reviews']['Review']
    for d in reviews:
        for s in d["sentences"]['sentence']:
            # register words
            if "@OutOfScope" not in s and "Opinions" in s and "text" in s:
                if isinstance(s["Opinions"]["Opinion"], list):
                    annotations = [o["@category"] for o in s["Opinions"]["Opinion"]]
                else:
                    annotations = [s["Opinions"]["Opinion"]["@category"]]
            if len(annotations) > 0:
                row = {}
                for k in label_kinds:
                    if k in annotations:
                        row[k] = 1
                    else:
                        row[k] = 0
                labels.append(row)
                sentences.append(s['text'])

labels = pd.DataFrame(labels)
print(labels.head(5))

import matplotlib.pyplot as plt

# labels.sum(axis=0).sort_values(ascending=False).plot.bar()
# plt.show()

if len(sentences) != len(labels):
    raise Exception("sentence and label count does not match!")

print("{} data is available.".format(len(labels)))

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, SGDClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([("vectorize", SentenceToVec(stop_words=stop_words)), ("clf", OneVsRestClassifier(LinearSVC(random_state=0)))])
# model = Pipeline([("vectorize", CountVectorizer(stop_words='english', analyzer='word', ngram_range=(1, 1),
#                                                 tokenizer=tokenize, max_features=700)),
#                   ("clf", OneVsRestClassifier(LinearSVC(random_state=0)))])
# model = Pipeline([("vectorize", SentenceToVec(stop_words)), ("clf", MLPClassifier(max_iter=20))])



from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit


# learning curve function
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(
        estimator, title, X, y, ylim=None, cv=None,
        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

X = np.array(sentences)
print(X.shape)
print(labels.shape)
plot_learning_curve(model, "Slot1 baseline learning curve ",
                    X, labels, ylim=(0.0, 1.01), cv=cv, n_jobs=1, verbose=4)

plt.show()


# vectorizer = CountVectorizer(stop_words='english', analyzer='word', ngram_range=(1, 2), max_features=700)
# my_vectorizer = SentenceToVec();
# vectorizer.fit_transform(sentences)
# print(vectorizer.get_feature_names())
# ...........predict..........
# model.fit(X, labels)

# import pickle
# # save the model to disk
# filename = 'finalized_model.sav'
# # pickle.dump(model, open(filename, 'wb'))
#
# # some time later...
#
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
#
# test_labels = []
# test_sentences = []
# with open("Data/ABSA15_Restaurants_Test.json") as j:
#     reviews = json.load(j)['Reviews']['Review']
#     for d in reviews:
#         for s in d["sentences"]['sentence']:
#             # register words
#             if "@OutOfScope" not in s and "Opinions" in s and "text" in s:
#                 tokenized = tokenize(s['text'])
#                 if isinstance(s["Opinions"]["Opinion"], list):
#                     annotations = [o["@category"] for o in s["Opinions"]["Opinion"]]
#                 else:
#                     annotations = [s["Opinions"]["Opinion"]["@category"]]
#             if len(annotations) > 0:
#                 row = {}
#                 for k in label_kinds:
#                     if k in annotations:
#                         row[k] = 1
#                     else:
#                         row[k] = 0
#                 test_labels.append(row)
#                 test_sentences.append(tokenized)
#
# test_labels = pd.DataFrame(test_labels)
# X_test = np.array(test_sentences)
# print(len(test_labels))
# print(X_test.shape)
#
# result = loaded_model.score(X_test, test_labels)
# print(result)