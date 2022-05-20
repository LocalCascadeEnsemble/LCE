LCE Documentation
====================

**Local Cascade Ensemble (LCE)** is a machine learning method that **further enhances** the prediction performance of the state-of-the-art **Random Forest** and **XGBoost**.
LCE combines their strengths and adopts a complementary diversification approach to obtain a better generalizing predictor.
Specifically, LCE is a hybrid ensemble method that combines an explicit boosting-bagging approach to handle the bias-variance trade-off faced by 
machine learning models and an implicit divide-and-conquer approach to individualize classifier errors on different parts of the training data.
LCE has been evaluated on a public benchmark and published in the journal *Data Mining and Knowledge Discovery*.

LCE package is **compatible with scikit-learn**; it passes the `check_estimator <https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator>`_.
Therefore, it can interact with scikit-learn pipelines and model selection tools.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial

   auto_examples/index
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   reference

`Getting started <quick_start.html>`_
-------------------------------------

Quick start tutorial showing snippets for you to try out LCE.

`Documentation <api.html>`_
-------------------------------

API documentation of LCE. 


`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples using LCE on public datasets.

`Reference <reference.html>`_
--------------------------------------

The full information about LCE can be found in the associated `journal paper <https://hal.inria.fr/hal-03599214/document>`_:

.. [1] Fauvel, K., E. Fromont, V. Masson, P. Faverdin and A. Termier. "XEM: An explainable-by-design ensemble method for multivariate time series classification", Data Mining and Knowledge Discovery, 36(3):917–957, 2022