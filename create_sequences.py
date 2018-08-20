import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from SequenceClass import Sequences

ROOT = os.environ['HOME'] + '/python/Vincent_Project/'
RANDOM_STATE = 14
SEQ_LENGTH = 400
NO_BINS = 1000
training_size = 0.8
discrete_range = np.linspace(-0.00350000016251, 0.00350000016251, NO_BINS)
encoder = OneHotEncoder(n_values = 1 + NO_BINS, dtype = 'int8')


LFPs = pickle.load(open(ROOT + 'final_LFP_DF.p', 'rb'))
for shank_label, shank3 in LFPs.groupby(axis = 0, level = 'Shank3'):
    for ROI_label, ROI in shank3.groupby(axis = 0, level = 'ROI'):
        seqs = Sequences()
        seqs.header['RANDOM_STATE'] = RANDOM_STATE
        seqs.header['data_label'] = '_'.join([shank_label, ROI_label])
        seqs.header['sequence_length'] = SEQ_LENGTH

        noTrials = len(ROI)
        list_to_be_shuffled = np.arange(noTrials)
        np.random.seed(RANDOM_STATE)
        np.random.shuffle(list_to_be_shuffled)

        cutoff = int(np.floor(noTrials * training_size))
        train_index = list_to_be_shuffled[:cutoff]
        test_index = list_to_be_shuffled[cutoff:]
        assert len(train_index) + len(test_index) == noTrials

        TRAIN = ROI.iloc[train_index]
        TEST = ROI.iloc[test_index]
        assert len(TRAIN) + len(TEST) == noTrials

        #want to know that the split was more or less even
        TRAIN_ERRORS = np.sum([w for w in TRAIN.index.labels[6]])
        TRAIN_ERRORS = np.float(TRAIN_ERRORS) / len(TRAIN)
        TEST_ERRORS = np.sum([w for w in TEST.index.labels[6]])
        TEST_ERRORS = np.float(TEST_ERRORS) / len(TEST)
        print 'Ratio Errors Train:%.2f - Test:%.2f' %(TRAIN_ERRORS, TEST_ERRORS)

        X_train, y_train = split_sequences(TRAIN, SEQ_LENGTH)
        X_test, y_test = split_sequences(TEST, SEQ_LENGTH)
        assert  X_train.dtype == 'int8'
        assert  y_train.dtype == 'int8'
        assert  X_test.dtype == 'int8'
        assert  y_test.dtype == 'int8'

        seqs.X_train = X_train
        seqs.y_train = y_train
        seqs.X_test = X_test
        seqs.y_test = y_test

        pickle.dump(seqs,
            open(ROOT + 'Sequences/' + seqs.header['data_label'] + '.p','wb'))

def split_sequences(df, SEQ_LENGTH):
    noTrials = len(df)
    seqs_per_seq = int(np.floor(2001 / SEQ_LENGTH))
    X = np.zeros((noTrials * seqs_per_seq, SEQ_LENGTH))
    y = np.zeros((noTrials * seqs_per_seq, 1))
    for trial in range(noTrials):
        label = df.index.labels[6][trial]
        for tmp_seq in range(seqs_per_seq):
            X[trial * seqs_per_seq + tmp_seq,:] = \
                df.iloc[trial][SEQ_LENGTH * tmp_seq : SEQ_LENGTH * (tmp_seq + 1)]
            y[trial * seqs_per_seq + tmp_seq, 0] = label
    X = np.digitize(X, discrete_range)
    X = one_hot_encoding(X)
    y = y.astype('int8')
    return X, y

def one_hot_encoding(X):
    new_X = np.zeros([X.shape[0], X.shape[1], 1 + NO_BINS], dtype='int8')
    for trial in range(len(X)):
        new_X[trial,:,:] = \
                encoder.fit_transform(X[trial,:].reshape(-1,1)).todense()
    return new_X
