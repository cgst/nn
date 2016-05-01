import numpy as np
np.random.seed(333715)

from enum import Enum
import logging
import sys

from gensim.models import Word2Vec
from spacy.en import English
from keras.layers import Activation, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
import pandas as pd


class WordRepr(Enum):
    spacy = 1
    glove = 2


wr = WordRepr.glove


logging.info("Loading language models")
wv_model = Word2Vec.load(sys.argv[2])
nlp = English()


do_sample = False
batch_size = 32
sentence_len = 80
train_ratio = 0.75
nb_epoch = 15


def spacy_sentence(text):
    return [w.vector for w in nlp(text)[:sentence_len]]


def glove_sentence(text):
    global wv_model
    sentence = []
    for token in nlp(text)[:sentence_len]:
        word = token.text.lower()
        if word not in wv_model:
            continue
        sentence.append(wv_model[word])
    return sentence


def predict(text_sents, sent_function=glove_sentence):
    sents = list(map(sent_function, text_sents))
    x_pred = sequence.pad_sequences(sents, maxlen=sentence_len,
                                    dtype='float32')
    y_pred = [y[0] for y in model.predict(x_pred, verbose=1)]
    return y_pred


if wr is WordRepr.glove:
    model_dim = wv_model.syn0.shape[1]
    sent_function = glove_sentence
else:
    model_dim = nlp.vocab.vectors_length
    sent_function = spacy_sentence


logging.basicConfig(level=logging.DEBUG)

logging.info("Loading questions dataset")
questions = pd.read_csv(sys.argv[1])
phrases = questions['phrase'].map(sent_function)
x = sequence.pad_sequences(phrases, maxlen=sentence_len, dtype='float32')
y = questions["is_question"].map(float)
y = np.column_stack((y, 1.0 - y))

# Train/test datasets
if do_sample:
    x, y = x[:100], y[:100]
mask = np.random.rand(len(x)) < train_ratio
x_train, y_train = x[mask], y[mask]
x_test, y_test = x[~mask], y[~mask]
logging.info("Train shape %s, %s", x_train.shape, y_train.shape)
logging.info("Test shape %s, %s", x_test.shape, y_test.shape)

# Build network
model = Sequential()
model.add(LSTM(model_dim, input_shape=(sentence_len, model_dim,),
               dropout_W=0.2, dropout_U=0.2, activation='tanh'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(x_test, y_test))
score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
logging.info("Score: %.2f\tAccuracy: %.2f", score, accuracy)

# Save model
output_name = sys.argv[3]
open("{}.json".format(output_name), "w").write(model.to_json())
model.save_weights("{}.weights".format(output_name))
