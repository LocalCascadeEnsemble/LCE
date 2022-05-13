
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
   

**Local Cascade Ensemble (LCE)** is a machine learning method that **further enhances** the prediction performance of the state-of-the-art **Random Forest** and **XGBoost**. LCE combines their strengths and adopts a complementary diversification approach to obtain a better generalizing predictor. Specifically, LCE is a hybrid ensemble method that combines an explicit boosting-bagging approach to handle the bias-variance trade-off faced by machine learning models and an implicit divide-and-conquer approach to individualize classifier errors on different parts of the training data. LCE has been evaluated on a public benchmark and published in the journal *Data Mining and Knowledge Discovery*.

LCE package is **compatible with scikit-learn**; it passes the `check_estimator <https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator>`_. Therefore, it can interact with scikit-learn pipelines and model selection tools.


Installation
~~~~~~~~~~~~

LCE package can be installed using ``pip``::

	pip install lcensemble

or ``conda``::

	conda install -c conda-forge lcensemble


Documentation
~~~~~~~~~~~~~

LCE documentation can be found `here <https://lce.readthedocs.io/en/latest/>`_.


Reference
~~~~~~~~~

The full information about LCE can be found in the associated `journal paper <https://hal.inria.fr/hal-03599214/document>`_.

.. [1] Fauvel, K., E. Fromont, V. Masson, P. Faverdin and A. Termier. "XEM: An explainable-by-design ensemble method for multivariate time series classification", Data Mining and Knowledge Discovery, 2022