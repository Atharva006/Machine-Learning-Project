import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv('spam_ham_dataset.csv')
X = df["text"]
y = df["label_num"]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)

vectorizer = TfidfVectorizer(stop_words = "english")
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vector, y_train)

y_pred = model.predict(X_test_vector)


print("Accuracy", accuracy_score(y_test, y_pred))
print(classification_report(y_pred, y_test))

msg = ["You are selected as a winner. Call now to claim your free prize."]
msg_vector = vectorizer.transform(msg)


prediction = model.predict(msg_vector) [0]

if prediction == 1:
  print("spam")
else:
  print("Not spam")
