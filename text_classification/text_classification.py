# Text Classification using TF-IDF + Logistic Regression
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

texts = ["I love AI", "Hate this product", "Amazing work", "Very bad experience"]
labels = [1,0,1,0]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
