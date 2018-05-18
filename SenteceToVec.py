from collections import Counter

import nltk
import numpy as np
from nltk import PunktSentenceTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

class SentenceToVec(BaseEstimator, TransformerMixin):

    def __init__(self, stop_words, vector_len=1000):
        self.vocab = []
        self.stop_words = stop_words
        self.vector_len = vector_len

        self.tokenizer = PunktSentenceTokenizer()

    def format_word(self, word):
        if word.isdigit():
            return "0"
        elif word in self.stop_words:
            return ""
        else:
            return word.strip()

    def tokenize(self, sentence):
        res_tokens = []
        tokens_temp = self.tokenizer.tokenize(sentence)
        for tokens in tokens_temp:
            tokens = nltk.word_tokenize(tokens)
            tokens = [self.format_word(t) for t in tokens]
            res_tokens += [t for t in tokens if t]
        return res_tokens

    def fit(self, X, y=None):
        self.vocab = []
        word_freq = Counter()
        for i in range(X.shape[0]):
            for w in self.tokenize(X[i]):
                if w not in self.stop_words:
                    word_freq[w] += 1

        for term, freq in word_freq.most_common():
            if len(self.vocab) < self.vector_len:
                self.vocab.append(term)
        return self

    def _vectorize(self, words):
        freq = dict(Counter(words))
        vector = []
        for v in self.vocab:
            vector.append(freq[v] if v in words else 0)
        return np.array(vector)

    def transform(self, X, copy=True):
        _X = np.zeros((X.shape[0], len(self.vocab)))
        for i in range(X.shape[0]):
            _X[i] = self._vectorize(self.tokenize(X[i]))
        return _X

