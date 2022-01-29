"""Setup file for the label_flip_revised package

To install this local package
    python -m pip install .
To upgrade this package
    python -m pip install --upgrade .
"""
from setuptools import setup, find_packages

setup(
    name='label_flip_revised',
    description='Perform Label Flip Attack on a machine learning classifier',
    packages=find_packages(),
    version='0.0.1',
    python_requires='>=3.6',
    install_requires=[
        'adversarial-robustness-toolbox==1.9.1',
        'imbalanced-learn==0.9.0',
        'jupyterlab',
        'matplotlib',
        'pandas',
        'scikit-learn==1.0.2',
        'seaborn',
        'tqdm',
    ],
)
