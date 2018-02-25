import os
import gensim
import numpy
import tensorflow as tf

model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
print(model['good'])