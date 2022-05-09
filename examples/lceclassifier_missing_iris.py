"""
=================================================
LCEClassifier on Iris dataset with missing values
=================================================

An example of :class:`lce.LCEClassifier`
"""

import numpy as np
from lce import LCEClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Load data and generate a train/test split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

# Input 20% of missing values per variable in the train set
np.random.seed(0)
m = 0.2
for j in range(0, X_train.shape[1]):
    sub = np.random.choice(X_train.shape[0], int(X_train.shape[0]*m))
    X_train[sub, j] = np.nan

# Train LCEClassifier with default parameters
clf = LCEClassifier(n_jobs=-1, random_state=123)
clf.fit(X_train, y_train)

# Make prediction and generate classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))