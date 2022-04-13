#! /usr/bin/env python

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('lce', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'lce'
DESCRIPTION = 'LCE package'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_TYPE = 'text/x-rst'
MAINTAINER = 'Kevin Fauvel'
MAINTAINER_EMAIL = 'kfauvel.lce@gmail.com'
URL = 'https://lce.readthedocs.io/en/latest/'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/LocalCascadeEnsemble/LCE'
VERSION = __version__
INSTALL_REQUIRES = ['hyperopt', 'matplotlib', 'numpy', 'scikit-learn', 'xgboost']
PROJECT_URLS = {
    "Documentation": "https://lce.readthedocs.io/en/latest/",
}
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python :: 3'
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
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
      python_requires='>=3.6',
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
