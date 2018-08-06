import sys
import sutils.config
import h5py
import pandas as pd
import itertools
import numpy as np
import os
import pickle

ROOT = sutils.config.ROOT
mat_files_dir = ROOT + 'Vincent_Project/new_labels/'
dataframe_files_dir = ROOT + 'Vincent_Project/'

idx = pd.IndexSlice
Shank3 = ['WT', 'Het']
ROI = ['mPFC', 'vCA1', 'dCA1']
delay = [5, 30, 60]
trial_type = ['correct', 'error']
arm = ['GA', 'SA']




def parse_identity(string):

    ROI = ['mPFC', 'vCA1', 'dCA1']
    trial_type = ['correct', 'error']
    arm = ['GA', 'SA']

    if string.find('correct') >= 0:
        tt = 0
    elif string.find('error') >= 0:
        tt = 1
    else:
        tt = -1

    if string.find('_c_') >= 0:
        ga = 0
    elif string.find('_r_') >= 0:
        ga = 1
    else:
        ga = -1

    if string.find('mPFC') >= 0:
        roi = 0
    elif string.find('vCA1')>= 0:
        roi = 1
    elif string.find('dCA1')>= 0:
        roi = 2
    else:
        roi = -1

    if roi >= 0:
        d = int(string[-string[::-1].find('_'):])
    else:
        d = -1

    a = tt, roi, ga, d
    if all([w >= 0 for w in a]):
        tt = trial_type[tt]
        roi = ROI[roi]
        ga = arm[ga]
        return roi, ga, d, tt
    else:
        return None

giant_list = []
for dataset in os.listdir(mat_files_dir):
    if dataset.find('Het') > 0:
        filename_out = 'Het'
    elif dataset.find('WT') > 0:
        filename_out = 'WT'
    tmp = h5py.File(mat_files_dir + dataset)


    #first just scroll to find dimensions of dataset
    for k, v in tmp.items():
        #how many animals and how many sessions each animal did
        if k.find('listRat') >= 0:
            noAnimals = len(v)
            no_sessions_per_animal = v.value
            noSessions = int(np.sum(no_sessions_per_animal))
            correct_error_info = np.zeros([noSessions, 6])



    animals = ['A%i' %i for i in range(noAnimals)]
    animal_sessions_tuples = [('A%i' %lab, 'S%i' %v)
    for lab, w in enumerate(no_sessions_per_animal)
    for v in range(w)]

    for k, v in tmp.items():
        if k.find('list') >= 0 and k.find('Rat') < 0:
            if k.find('Correct') > 0:
                Correct = 1
            else:
                Correct = 0
            if k.find('5') >= 0:
                tmp_delay = 0
            elif k.find('30') >= 0:
                tmp_delay = 1
            elif k.find('60') >= 0:
                tmp_delay = 2
            correct_error_info[:, tmp_delay + 3 * Correct] = v.value.ravel()



    #fill in all values but still without animal and session information

    for k, v in tmp.items():
        a = parse_identity(k)
        if a:
            noTrials = v.shape[0]
            row_index = pd.MultiIndex.from_product(
                    [ROI, arm, delay, trial_type, range(noTrials)],
                    names =['ROI', 'Arm', 'Delay', 'trial_type', 'trial'])


            if a[2] == 5:
                delay_type = 0
            elif a[2] == 30:
                delay_type = 1
            elif a[2] == 60:
                delay_type = 2
            if a[3] == 'correct':
                tmp_trial_type = 1
            elif a[3] == 'error':
                tmp_trial_type = 0
            tmp_CEI = correct_error_info[:, delay_type + 3 * tmp_trial_type]
            assert sum(tmp_CEI) == v.shape[0]
            sub_tuples = \
            [ast + (bb,) for ast, b in zip(animal_sessions_tuples, tmp_CEI)
            for bb in range(int(b))]


            sub_tuples = [(filename_out,) + si + a for si in sub_tuples]


            sub_index = pd.MultiIndex.from_tuples(sub_tuples,
                        names = ['Shank3', 'Animal', 'Session',
                                 'trial', 'ROI', 'Arm', 'Delay', 'trial_type'])

            sub_df = pd.DataFrame(v.value, index = sub_index, columns = np.arange(2001))

            giant_list.append(sub_df)


DF = pd.concat(giant_list)
DF.sort_index(axis = 0, inplace = True)

DF = DF.swaplevel('trial','trial_type')
DF = DF.swaplevel('trial_type','ROI')
DF = DF.swaplevel('trial_type','Arm')
DF = DF.swaplevel('trial_type','Delay')

DF.sort_index(axis = 0, inplace = True)

#check all the sizes match what they should
for dataset in os.listdir(mat_files_dir):
    if dataset.find('Het') > 0:
        filename_out = 'Het'
    elif dataset.find('WT') > 0:
        filename_out = 'WT'
    tmp = h5py.File(mat_files_dir + dataset)
    for k, v in tmp.items():
        a = parse_identity(k)
        if a:
            assert v.shape == DF.loc[
                idx[filename_out, :, :, a[0], a[1], a[2], a[3]], :].shape


pickle.dump(DF, open(dataframe_files_dir + 'final_LFP_DF.p', 'wb'))
