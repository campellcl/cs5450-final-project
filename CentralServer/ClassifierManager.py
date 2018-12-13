"""
ClassifierManager.py
Interprets and responds to all classification requests received (and then relayed) by the central server. This class
can be queried as to the state of the available classifiers. This class can be queried for the hyperparamteres and
evaluation metrics associated with each classifier. This class can also be used to perform classifications, as this
class will handle any conversions necessary to get the provided image fed into the respective model.
"""

import os
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class ClassifierManager:
    classifiers = None
    trained_classifiers = None

    def __init__(self, trained_classifiers=None):
        if trained_classifiers is not None:
            self.trained_classifiers = trained_classifiers

    def get_list_of_classifiers(self):
        raise NotImplementedError

    def get_list_of_trained_classifiers(self):
        raise NotImplementedError

    def add_trained_model(self, model_path):
        raise NotImplementedError



if __name__ == '__main__':
    classifier_manager = ClassifierManager()
