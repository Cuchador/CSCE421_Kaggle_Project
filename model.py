import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Model():
    def __init__(self, args):
        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguments to the initialization as needed
        self.model = RandomForestClassifier(**args)
        ########################################################################

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        ############################ Your Code Here ############################
        # Fit your model to the training data here
        self.model.fit(x_train, y_train)
        ########################################################################

    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortality for each x
        return self.model.predict_proba(x)
        ########################################################################
