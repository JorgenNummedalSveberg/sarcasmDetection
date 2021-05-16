from nltk.lm import MLE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams
from nltk import TweetTokenizer
import nltk
# import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pydot

nltk.download('stopwords')
class Comment:
    def __init__(self, sarcastic, comment):
        self.comment = comment
        self.sarcastic = sarcastic




# columns = pd.read_csv('train-balanced-sarcasm.csv')
# columns = pd.read_csv('OnionOrNot.csv')
# comments = [Comment(label, str(comment)) for (comment, label) in zip(columns["text"], columns["label"])]
columns = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)
# comments = [*comments, *[Comment(label, str(comment)) for (comment, label) in zip(columns["headline"], columns["is_sarcastic"])]]
comments = [Comment(label, str(comment)) for (comment, label) in zip(columns["headline"], columns["is_sarcastic"])]

for item in comments:
    item.comment.replace('â€™', "'")

# comments = comments[0:10000]

train, test = train_test_split(comments, test_size=0.1)
child_train = np.array([item.comment for item in train])
child_test = np.array([item.comment for item in test])
label_train = np.array([item.sarcastic for item in train])
label_test = np.array([item.sarcastic for item in test])

vocab_size = 5000
embedding_dim = 32
oov_tok = "<oov>"

child_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
child_tokenizer.fit_on_texts(child_train)

vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=20)
vectorizer.adapt(child_train)

child_index = child_tokenizer.word_index
max_length = 30
trunc_type = 'post'
padding_type = 'post'
epochs = 5

child_train_seq = child_tokenizer.texts_to_sequences(child_train)
child_train_padded = pad_sequences(child_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

child_test_seq = child_tokenizer.texts_to_sequences(child_test)
child_test_padded = pad_sequences(child_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=5, min_lr=0.001)
             ]
model = keras.models.Sequential([
    vectorizer,
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    layers.Dense(16, activation='relu'),

    # keras.layers.GlobalAveragePooling1D(),
    # keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

keras.utils.plot_model(model, "NoContextEmbeddingBidirectional.png", show_shapes=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(child_train, label_train, epochs=epochs, callbacks=callbacks)
# history = model.fit(child_train_padded, label_train, epochs=epochs)


train_loss, train_accuracy = model.evaluate(child_train, label_train, verbose=2)
test_loss, test_accuracy = model.evaluate(child_test, label_test, verbose=2)
# train_loss, train_accuracy = model.evaluate(child_train_padded, label_train, verbose=2)
# test_loss, test_accuracy = model.evaluate(child_test_padded, label_test, verbose=2)

