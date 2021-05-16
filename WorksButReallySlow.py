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


# specify GPU device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# n_gpu = torch.cuda.device_count()
# print(n_gpu)
# print(torch.cuda.get_device_name(0))



# attempt at using a different dataset
# sarcolumns = pd.read_csv('SPIRS-sarcastic.csv')
# nonsarcolumns = pd.read_csv('SPIRS-non-sarcastic.csv')
# lengthsar = [0] * len(list(sarcolumns)[0])
# lengthnonsar = [0] * len(list(nonsarcolumns)[0])
# sarcomments = [Comment(1, str(comment)) for comment in sarcolumns["sar_text"] if comment != "" and comment != 'nan']
# nonsarcomments = [Comment(0, str(comment)) for comment in nonsarcolumns["sar_text"] if comment != "" and comment != 'nan']
# comments = [*sarcomments, *nonsarcomments]
# random.shuffle(comments)

columns = pd.read_csv('train-balanced-sarcasm.csv')
comments = [Comment(label, str(comment), str(parent)) for (comment, label, parent) in zip(columns["comment"], columns["label"], columns["parent_comment"])]

for item in comments:
    item.comment.replace('â€™', "'")

comments = comments[0:10000]

train, test = train_test_split(comments, test_size=0.1)
child_train = np.array([item.comment for item in train])
child_test = np.array([item.comment for item in test])
parent_train = np.array([item.parent for item in train])
parent_test = np.array([item.parent for item in test])
label_train = np.array([item.sarcastic for item in train])
label_test = np.array([item.sarcastic for item in test])

vocab_size = 10000
embedding_dim = 16
oov_tok = "<oov>"

child_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
parent_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
child_tokenizer.fit_on_texts(child_train)
parent_tokenizer.fit_on_texts(parent_train)

child_index = child_tokenizer.word_index
parent_index = parent_tokenizer.word_index
max_length = 100
trunc_type = 'post'
padding_type = 'post'
epochs = 2

child_train_seq = child_tokenizer.texts_to_sequences(child_train)
child_train_padded = pad_sequences(child_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

child_test_seq = child_tokenizer.texts_to_sequences(child_test)
child_test_padded = pad_sequences(child_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

parent_train_seq = parent_tokenizer.texts_to_sequences(parent_train)
parent_train_padded = pad_sequences(parent_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

parent_test_seq = parent_tokenizer.texts_to_sequences(parent_test)
parent_test_padded = pad_sequences(parent_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)


input1 = keras.Input(shape=(1,), dtype=tf.string, name='parent')
input2 = keras.Input(shape=(1,), dtype=tf.string, name='child')

child_vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=20)
child_vectorizer.adapt(child_train)
parent_vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=20)
parent_vectorizer.adapt(parent_train)

# about 100 train 61 test
# x1 = parent_vectorizer(input1)
# x2 = keras.layers.Embedding(vocab_size, 64)(x1)
# x3 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x2)
# x4 = keras.layers.Bidirectional(keras.layers.LSTM(32))(x3)
# x5 = keras.layers.Dense(64, activation='relu')(x4)
# x6 = keras.layers.Dropout(0.5)(x5)
# x7 = keras.layers.Dense(1)(x6)
#
# y1 = child_vectorizer(input2)
# y2 = keras.layers.Embedding(vocab_size, 64)(y1)
# y3 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(y2)
# y4 = keras.layers.Bidirectional(keras.layers.LSTM(32))(y3)
# y5 = keras.layers.Dense(64, activation='relu')(y4)
# y6 = keras.layers.Dropout(0.5)(y5)
# y7 = keras.layers.Dense(1)(y6)

# about 100 train and 62 test
# x1 = parent_vectorizer(input1)
# x2 = keras.layers.Embedding(vocab_size, 32)(x1)
# x3 = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(x2)
# x4 = keras.layers.Bidirectional(keras.layers.LSTM(16))(x3)
# x5 = keras.layers.Dense(32, activation='relu')(x4)
# x6 = keras.layers.Dropout(0.5)(x5)
# x7 = keras.layers.Dense(1)(x6)
#
# y1 = child_vectorizer(input2)
# y2 = keras.layers.Embedding(vocab_size, 32)(y1)
# y3 = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(y2)
# y4 = keras.layers.Bidirectional(keras.layers.LSTM(16))(y3)
# y5 = keras.layers.Dense(32, activation='relu')(y4)
# y6 = keras.layers.Dropout(0.5)(y5)
# y7 = keras.layers.Dense(1)(y6)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# about 100 train and 63 test
x1 = parent_vectorizer(input1)
x2 = keras.layers.Embedding(vocab_size, 32)(x1)
x3 = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(x2)
x4 = keras.layers.Dropout(0.5)(x3)
x5 = keras.layers.Bidirectional(keras.layers.LSTM(16))(x4)
x6 = keras.layers.Dropout(0.5)(x5)
x7 = keras.layers.Dense(16, activation='relu')(x6)
x8 = keras.layers.Dropout(0.5)(x7)

y1 = child_vectorizer(input2)
y2 = keras.layers.Embedding(vocab_size, 32)(y1)
y3 = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(y2)
y4 = keras.layers.Dropout(0.5)(y3)
y5 = keras.layers.Bidirectional(keras.layers.LSTM(16))(y4)
y6 = keras.layers.Dropout(0.5)(y5)
y7 = keras.layers.Dense(16, activation='relu')(y6)
y8 = keras.layers.Dropout(0.5)(y7)


outputs = layers.Dense(1, activation='sigmoid')(layers.concatenate([x8, y8]))

model = keras.Model(
    inputs=[input1, input2],
    outputs=[outputs]
)

# If you want a model of the AI model
# keras.utils.plot_model(model, "ContextEmbeddingBidirectional.png", show_shapes=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit({'parent': parent_train, 'child': child_train}, label_train, epochs=epochs, verbose=2, callbacks=[callback])

train_loss, train_accuracy = model.evaluate({'parent': parent_train, 'child': child_train}, label_train, verbose=2)
test_loss, test_accuracy = model.evaluate({'parent': parent_test, 'child': child_test}, label_test, verbose=2)


print("Train loss:", train_loss, "Train accuracy:", train_accuracy)
print("Test loss:", test_loss, "Test accuracy:", test_accuracy)
