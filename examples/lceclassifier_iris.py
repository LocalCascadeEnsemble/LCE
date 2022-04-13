"""
=============================
LCEClassifier on Iris dataset
=============================

An example of :class:`lce.LCEClassifier`
"""

from lce import LCEClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Load data and generate a train/test split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

# Train LCEClassifier with default parameters
clf = LCEClassifier(random_state=0)
clf.fit(X_train, y_train)

# Make prediction and generate classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))