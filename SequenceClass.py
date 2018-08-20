
import os
import copy
import pickle
import numpy as np
import pandas as pd
idx = pd.IndexSlice


class Sequences:
    def __init__(self):
        self.header = {'sequence_length': 0,
                       'RANDOM_STATE' : 6,
                       'data_label' : ''}

        self.X_train = []
        self.y_train = []
        self.X_validate = []
        self.y_validate = []
        self.X_test = []
        self.y_test = []
