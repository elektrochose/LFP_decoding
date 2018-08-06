#THIS IS EPSILON ROOT DIRECTORY !
ROOT = '/home/pablo/python/'
#standard modules
import sys
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
import numpy as np
import os
import pickle
import time
import itertools
import multiprocessing as mp

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


RANDOM_STATE = 7
test_size = 0.2


dataframe_files_dir = ROOT + 'Vincent_Project/python_data/'
model_dir = ROOT + 'Models/LSTM/Vincent'

idx = pd.IndexSlice
trial_type = ['correct', 'error']
ROI = ['mPFC', 'vCA1', 'dCA1']
arm = ['GA', 'SA']
fileNames = ['Het_discrete.p', 'WT_discrete.p']
hidden_dimensions = [2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]




def train_network(HD, dataset_path, ROI, arm, model_path):
    if not os.path.isfile(model_path):
        print 'training %s' %model_path
        df = pickle.load(open(dataset_path, 'rb'))
        tmp_df = df.loc[idx[ROI, :, arm, :], :]
        noTrials = len(tmp_df)
        y = pd.Series(np.zeros(noTrials, dtype=int), index = tmp_df.index)
        y.loc[idx[:,'correct',:,:]] = 1

        print 'separating train/test split'
        #shuffling data and determining train/test split
        shuffleIndex = np.arange(noTrials)
        np.random.seed(RANDOM_STATE)
        np.random.shuffle(shuffleIndex)
        cutoff = int(np.floor(noTrials * (1 - test_size)))

        #doing the actual split
        X_train = tmp_df.iloc[shuffleIndex[:cutoff]]
        y_train = y.iloc[shuffleIndex[:cutoff]]
        X_test = tmp_df.iloc[shuffleIndex[cutoff:]]
        y_test = y.iloc[shuffleIndex[cutoff:]]

        print 'creating model'
        # create the model
        model = Sequential()
        model.add(Embedding(1001, 256, input_length=2001))
        model.add(LSTM(HD))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs = 30, batch_size=64)
        model.save(model_path)

pool = mp.Pool(processes = 32)

for fileName, roi, maze_arm, hd in \
    itertools.product(fileNames, ROI, arm, hidden_dimensions):
    dataset_path = dataframe_files_dir + fileName
    model_path = \
    '/'.join([model_dir,
              fileName[:fileName.find('_')],
              roi,
              maze_arm,
              'LSTM' + str(hd) + '.h5'])

    pool.apply_async(train_network,
                    [hd, dataset_path, roi, maze_arm, model_path])
