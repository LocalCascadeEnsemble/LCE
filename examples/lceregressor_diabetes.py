"""
================================
LCERegressor on Diabetes dataset
================================

An example of :class:`lce.LCERegressor`
"""

from lce import LCERegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Load data and generate a train/test split
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

# Train LCERegressor with default parameters
reg = LCERegressor(n_jobs=-1, random_state=0)
reg.fit(X_train, y_train)

# Make prediction 
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))