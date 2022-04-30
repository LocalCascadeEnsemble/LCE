#! /usr/bin/env python

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('lce', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'lcensemble'
DESCRIPTION = 'Local Cascade Ensemble package'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_TYPE = 'text/x-rst'
MAINTAINER = 'Kevin Fauvel'
MAINTAINER_EMAIL = 'kfauvel.lce@gmail.com'
URL = 'https://lce.readthedocs.io/en/latest/'
LICENSE = 'Apache-2.0'
DOWNLOAD_URL = 'https://github.com/LocalCascadeEnsemble/LCE'
VERSION = __version__
INSTALL_REQUIRES = ['hyperopt>=0.2.7', 'matplotlib', 'numpy>=1.19.2', 'pandas==1.3.5', 'scikit-learn>=1.0.0', 'xgboost==1.5.0']
PROJECT_URLS = {
    "Documentation": "https://lce.readthedocs.io/en/latest/",
}
CLASSIFIERS = ['License :: OSI Approved :: Apache Software License',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               'Programming Language :: Python :: 3.10',
               'Operating System :: OS Independent'
               ]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'pillow'
    ]
}

setup(name=DISTNAME,
      python_requires='>=3.7',
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      project_urls=PROJECT_URLS,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_TYPE,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
