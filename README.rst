LCE: Local Cascade Ensemble
===========================

|CircleCI|_ |ReadTheDocs|_ |PyPIversion|_ |PyPIpythonversion|_ |License|_ |DOI|_

.. |CircleCI| image:: https://circleci.com/gh/LocalCascadeEnsemble/LCE/tree/main.svg?style=shield
.. _CircleCI: https://circleci.com/gh/LocalCascadeEnsemble/LCE/tree/main
   
.. |ReadTheDocs| image:: https://readthedocs.org/projects/lce/badge/?version=latest
.. _ReadTheDocs: https://lce.readthedocs.io/en/latest/?badge=latest

.. |PyPIversion| image:: https://badge.fury.io/py/lcensemble.svg
.. _PyPIversion: https://pypi.python.org/pypi/lcensemble/

.. |PyPIpythonversion| image:: https://img.shields.io/pypi/pyversions/lcensemble.svg
.. _PyPIpythonversion: https://pypi.python.org/pypi/lcensemble/

.. |License| image:: https://img.shields.io/github/license/LocalCascadeEnsemble/LCE.svg
.. _License: https://pypi.python.org/pypi/lcensemble/

.. |DOI| image:: https://zenodo.org/badge/DOI/10.1007/s10618-022-00823-6.svg
.. _DOI: https://doi.org/10.1007/s10618-022-00823-6
   
.. raw:: html
	
	<p align="center">
	<img src="./logo/logo_lce.svg" width="25%">	
	</p>
   

**Local Cascade Ensemble (LCE)** is a machine learning method that **further enhances** the prediction performance of the state-of-the-art **Random Forest** and **XGBoost**.
LCE combines their strengths and adopts a complementary diversification approach to obtain a better generalizing predictor.
Specifically, LCE is a hybrid ensemble method that combines an explicit boosting-bagging approach to handle the bias-variance trade-off faced by 
machine learning models and an implicit divide-and-conquer approach to individualize classifier errors on different parts of the training data.
LCE has been evaluated on a public benchmark and published in the journal *Data Mining and Knowledge Discovery*.

LCE package is **compatible with scikit-learn**; it passes the `check_estimator <https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator>`_.
Therefore, it can interact with scikit-learn pipelines and model selection tools.


Getting Started
===============

Installation
------------

You can install LCE from `PyPI <https://pypi.org/project/lcensemble/>`_ with ``pip``::

	pip install lcensemble

or ``conda``::

	conda install -c conda-forge lcensemble
	

First Example on Iris Dataset
-----------------------------

LCEClassifier prediction on an Iris test set::

	from lce import LCEClassifier
	from sklearn.datasets import load_iris
	from sklearn.metrics import classification_report
	from sklearn.model_selection import train_test_split


	# Load data and generate a train/test split
	data = load_iris()
	X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

	# Train LCEClassifier with default parameters
	clf = LCEClassifier(n_jobs=-1, random_state=0)
	clf.fit(X_train, y_train)

	# Make prediction and generate classification report
	y_pred = clf.predict(X_test)
	print(classification_report(y_test, y_pred))


Documentation
=============
LCE documentation can be found `here <https://lce.readthedocs.io/en/latest/>`_.


Reference
=========
The full information about LCE can be found in the associated `journal paper <https://hal.inria.fr/hal-03599214/document>`_. 
If you use the package, please cite us with the following BibTex::

	@article{Fauvel22-LCE,
	  author = {Fauvel, K. and E. Fromont and V. Masson and P. Faverdin and A. Termier},
	  title = {{XEM: An Explainable-by-Design Ensemble Method for Multivariate Time Series Classification}},
	  journal = {Data Mining and Knowledge Discovery},
	  year = {2022},
	  doi = {10.1007/s10618-022-00823-6}
	}