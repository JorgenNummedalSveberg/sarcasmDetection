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
    def __init__(self, sarcastic, comment, parent):
        self.comment = comment
        self.sarcastic = sarcastic
        self.parent = parent




columns = pd.read_csv('train-balanced-sarcasm.csv')
comments = [Comment(label, str(comment), str(parent)) for (comment, label, parent) in zip(columns["comment"], columns["label"], columns["parent_comment"])]


for item in comments:
    item.comment.replace('â€™', "'")


comments = comments[0:10000]

train, test = train_test_split(comments, test_size=0.2)
child_train = np.array([item.comment for item in train])
child_test = np.array([item.comment for item in test])
parent_train = np.array([item.parent for item in train])
parent_test = np.array([item.parent for item in test])
label_train = np.array([item.sarcastic for item in train])
label_test = np.array([item.sarcastic for item in test])

vocab_size = 10000
# 32 worked better here than 16 and 64
embedding_dim = 32
oov_tok = "<oov>"

child_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
parent_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
child_tokenizer.fit_on_texts(child_train)
parent_tokenizer.fit_on_texts(parent_train)

child_index = child_tokenizer.word_index
parent_index = parent_tokenizer.word_index
max_length = 150
trunc_type = 'post'
padding_type = 'post'
epochs = 5
child_train_seq = child_tokenizer.texts_to_sequences(child_train)
child_train_padded = pad_sequences(child_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

child_test_seq = child_tokenizer.texts_to_sequences(child_test)
child_test_padded = pad_sequences(child_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

parent_train_seq = parent_tokenizer.texts_to_sequences(parent_train)
parent_train_padded = pad_sequences(parent_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

parent_test_seq = parent_tokenizer.texts_to_sequences(parent_test)
parent_test_padded = pad_sequences(parent_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)


input1 = keras.Input(shape=max_length, name='parent')
input2 = keras.Input(shape=max_length, name='child')


# x1 = layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(input1)
# x2 = layers.GlobalAveragePooling1D()(x1)
# x3 = layers.Dense(16, activation='relu')(x2)
#
# y1 = layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(input2)
# y2 = layers.GlobalAveragePooling1D()(y1)
# y3 = layers.Dense(16, activation='relu')(y2)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

x1 = layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(input1)
x2 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x1)
x3 = keras.layers.Bidirectional(keras.layers.LSTM(32))(x2)
x4 = layers.Dense(16, activation='relu')(x3)

y1 = layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(input2)
y2 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(y1)
y3 = keras.layers.Bidirectional(keras.layers.LSTM(32))(y2)
y4 = layers.Dense(16, activation='relu')(y3)


outputs = layers.Dense(1, activation='sigmoid')(layers.concatenate([x4, y4]))

model = keras.Model(
    inputs=[input1, input2],
    outputs=[outputs]
)
keras.utils.plot_model(model, "EmbeddingBidirectional.png", show_shapes=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit({'parent': parent_train_padded, 'child': child_train_padded}, label_train, epochs=epochs, verbose=2, callbacks=[callback])
# history = model.fit({'parent': parent_train_padded, 'child': child_train_padded}, label_train, epochs=epochs, validation_data=({'parent': parent_test_padded, 'child': child_test_padded}, label_test), verbose=2)

train_loss, train_accuracy = model.evaluate({'parent': parent_train_padded, 'child': child_train_padded}, label_train, verbose=2)
test_loss, test_accuracy = model.evaluate({'parent': parent_test_padded, 'child': child_test_padded}, label_test, verbose=2)

print("Train loss:", train_loss, "Train accuracy:", train_accuracy)
print("Test loss:", test_loss, "Test accuracy:", test_accuracy)

