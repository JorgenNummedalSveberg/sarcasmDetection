import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


class Comment:
    def __init__(self, sarcastic, comment):
        self.comment = comment
        self.sarcastic = sarcastic

# columns = pd.read_csv('train-balanced-sarcasm.csv')
columns = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)
# comments = [*comments, *[Comment(label, str(comment)) for (comment, label) in zip(columns["headline"], columns["is_sarcastic"])]]
comments = [Comment(label, str(comment)) for (comment, label) in zip(columns["headline"], columns["is_sarcastic"])]
comments = comments[0:100000]

vectorizer = CountVectorizer(max_features=500)
X = vectorizer.fit_transform([item.comment for item in comments])
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), [item.sarcastic for item in comments], test_size=0.5, random_state=0)
gnb = BernoulliNB()
pred = gnb.fit(X_train, y_train)
y_pred = pred.predict(X_test)
print("Accurate with ", (y_test == y_pred).sum()*100/X_test.shape[0], "%")

