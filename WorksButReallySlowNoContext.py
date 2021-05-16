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
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

nltk.download('stopwords')


class Comment:
    def __init__(self, sarcastic, comment):
        self.comment = comment
        self.sarcastic = sarcastic

    # def __str__(self):
    #     return "comment:", columns[1][index], ";parent:", columns[9][index]


# specify GPU device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# n_gpu = torch.cuda.device_count()
# print(n_gpu)
# print(torch.cuda.get_device_name(0))


columns = pd.read_csv('train-balanced-sarcasm.csv')
length = [0] * len(list(columns)[0])
comments = [Comment(label, str(comment)) for (comment, label) in zip(columns["comment"], columns["label"])]
# string_comments = [str(item.sarcastic) + ";" + item.comment for item in comments]
comments = comments[0:10000]

tokenized = [' '.join(nltk.sent_tokenize(item.comment)) for item in comments]

# cleaned_tokenized = tokenized
# stop_words = set(nltk.corpus.stopwords.words("english"))
# for word in stop_words:
#     for sent in cleaned_tokenized:
#         sent.replace(word, '')

# ps = nltk.stem.PorterStemmer()
# stemmed = [ps.stem(w) for w in cleaned_tokenized]
# print(stemmed)
X_train, X_test, y_train, y_test = train_test_split([item for item in tokenized], [item.sarcastic for item in comments], test_size=0.1)
vocab_size = 500

vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=20)
vectorizer.adapt(X_train)

# inputs = tf.keras.layers.Input(shape=(1, ), dtype=tf.string, name="text")
# outputs = vectorizer(inputs)
# model = tf.keras.Model(inputs, outputs)
# model.predict(["You don't really believe in the moon do you?"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
             ]

model = keras.models.Sequential([
    vectorizer,
    keras.layers.Embedding(vocab_size, 32),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1),
    # keras.layers.LSTM(32, activation='relu'),
    # keras.layers.Dense(32, activation='relu'),
    # keras.layers.Flatten(),
    # # keras.layers.LSTM(8, activation='relu'),
    # # keras.layers.Dense(8, activation='relu'),
    # # keras.layers.Flatten(),
    # keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], callbacks=callbacks)

model.fit(X_train, y_train, 64, epochs=10, verbose=2)
train_loss, train_accuracy = model.evaluate(X_train, y_train)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Train loss:", train_loss, "Train accuracy:", train_accuracy)
print("Test loss:", test_loss, "Test accuracy:", test_accuracy)

# Attempt 1

# vectorizer = CountVectorizer(max_features=500)
# X = vectorizer.fit_transform([item.comment for item in comments])
# X = vectorizer.fit_transform([item.comment for item in comments])
# X_train, X_test, y_train, y_test = train_test_split(X.toarray(), [item.sarcastic for item in comments], test_size=0.5, random_state=0)
# gnb = BernoulliNB()
# pred = gnb.fit(X_train, y_train)
# y_pred = pred.predict(X_test)
# print("Accurate with ", (y_test == y_pred).sum()*100/X_test.shape[0], "%")

# while True:
#     inptext = input("text: ")
#     testPred = pred.predict(vectorizer.transform([inptext]).toarray())
#     if testPred[0] == 1:
#         print("Sarcastic")
#     else:
#         print("Serious")
#     print()

# for item in string_comments:
#     print(item)
#     print()
# comment_tokens = [nltk.casual_tokenize(item.comment) for item in comments]
# print(comment_tokens)
# word_tokens = nltk.sent_tokenize([item.comment for item in comments])

# # Makes the language model. You can decide what size ngram you want it to train on, and how many tweets you want to use
# # There are quite a few tweets, so limiting this is necessary if you don't want to wait that long.
# # Somewhere around 10k will be decently quick, but will either make less sense or string together actual tweets
# # Somewhere around 50k will give you sentences that make more sense
# def fit(n, size=0):
#     trump_tweets = list(pd.read_json('trump.json', encoding='utf-8')['text'])
#     if size == 0:
#         size = len(trump_tweets)
#     trump_tweets = trump_tweets[:min(len(trump_tweets), size)]
#     trump_tweets = [TweetTokenizer().tokenize(sent) for sent in trump_tweets]
#     trump_tweets, trump_vocab = padded_everygram_pipeline(n, trump_tweets)
#     lm = MLE(n)
#     lm.fit(trump_tweets, trump_vocab)
#     return lm
#
#
# # Generates the sentence
# # It capitalizes words on the start of sentences, and continues to add words until it adds a period
# # It has some hard coded edits, like not having spaces before punctuation
# # as well as removing whitespaces from a few special characters
# def generate_sent(seed, lm):
#     first = False
#     orig_length = len(seed.split())
#     while True:
#         # Generates a new word based on the entire sentence so far
#         new_word = lm.generate(text_seed=seed.split(), num_words=1)
#         # Sorts out the </s> and <s> slots
#         if new_word == '<s>':
#             continue
#         # Seems to end up on a loop with </s> sometimes and doesn't want to make another sentence.
#         # Adds punctuation at end in case we hit a </s>
#         if new_word == '</s>':
#             last = seed[:-1]
#             last = nltk.word_tokenize(last)
#             last, last_tag = nltk.pos_tag(last)[0]
#             if last_tag == '.':
#                 break
#             if last_tag == ',':
#                 seed = seed[:-1]
#             seed += '.'
#             break
#
#         # Taggs the word for further processing, only one word so take first in array
#         new_word = nltk.word_tokenize(new_word)
#         word, tag = nltk.pos_tag(new_word)[0]
#
#         # If punctuation, add without space before
#         if tag == '.' or tag == ',':
#             # Don't add , if first
#             if tag == ',' and first:
#                 continue
#             seed += word
#             # Looks a bit weird, but not really that important. Just makes sure , doesn't flag first.
#             if tag == '.':
#                 first = True
#             # If period, end tweet, unless it added a period to a single word.
#             if word == '.' and len(seed.split()) > orig_length:
#                 break
#         # Else add word with space, if first, capitalize
#         # If first coule be wherever but no point in capitalizing punctuation
#         else:
#             if first:
#                 word = word.capitalize()
#                 first = False
#             seed += ' ' + word
#     seed = seed.replace(' ’ ', '’')
#     seed = seed.replace('“ ', '“')
#     seed = seed.replace(' ”', '”')
#     return seed
#
#
# # Fit to 10k tweets, 5-grams
# model = fit(5, 10000)
# # Some seeds
# print(generate_sent('The Mexicans', model))
# print(generate_sent('Make America', model))
# print(generate_sent('Obama', model))
