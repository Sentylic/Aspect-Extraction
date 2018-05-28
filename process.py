import gensim

# Load Google's pre-trained Word2Vec model.
print('start loading wordToVec..............')
wordToVec_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
print('finish loading wordToVec..............')
