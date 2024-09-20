import pandas as pd
import numpy as np
import random
import json
from collections import defaultdict

import sys
import os
sys.path.insert(1, os.path.dirname(__file__)+'/../Pipeline')
from Frames import get_last

random.seed(42)

csv_path = 'table.csv'
split_sets = json.load(open(os.path.dirname(__file__)+'/split_configs.json'))

# Delete column from annotation table with name (str)
def delete(name):
    df = pd.read_csv(csv_path, index_col=0)
    del df[name]
    df.to_csv(csv_path)
    
# Add split column to annotation table with name (str) using _ratios {'train':0.7 ...}
# Fill base_name with an existing split column to keep complete it to reach _ratios
# Fill pid_test (list of pid) to set them to test and complete split to reach _ratios
def add(name, _ratios, base_name='', pid_test=[]):
    
    # Annotation table
    df = pd.read_csv(csv_path, index_col=0)

    # Check if last_55 frame is available
    last_55 = get_last(1)

    # Select eyes index in i_good of not excluded eyes with available annotation for each item
    # Initialiase df[name] with 'new','excluded' according to i_good
    i_good, new = [], []
    for i, key in enumerate(df['key']): # not excluded eyes
        confoundings = not split_sets['include_confoundings'] and df['confounding factors'][i]
        last_55_missing = split_sets['need_last_55'] and last_55('', key=key)==[]
        if key.endswith('_Ex') or confoundings or last_55_missing:
            new += ['excluded']
        else:
            i_good += [i]
            new += ['new']
    for item in split_sets['items']: # available annotation for each item
        for i,answer in enumerate(df[item]):
            if answer in ['na', 'not assessable'] and i in i_good:
                i_good.remove(i)
                new[i] = 'excluded'
    df[name] = new # initialise df[name] 

    # Set pid_test to 'test' in df[name] and consider name as base_name split to complete
    if len(pid_test):
        if base_name: # combined argument not supported
            print('bad args: base_name + pid_test')
            return
        for i in i_good:
            if df['PID'][i] in pid_test:
                df.loc[i, name] = 'test'
        base_name = name # consider name as base_name split to complete
    
    # Separate eyes index between i_new/i_base and create split_ratio to reach _ratios
    split_ratio = _ratios.copy()
    if base_name:
        i_new = [i for i in i_good if df[base_name][i]=='new']
        i_base = [i for i in i_good if df[base_name][i]!='new']
        new = list(df[base_name])
        n = len(i_good)
        for group in split_ratio: # update split_ratio to reach _ratios
            split_ratio[group] = max(split_ratio[group]*n-new.count(group),0)
        n_new = sum([split_ratio[group] for group in split_ratio])
        split_ratio = {group:split_ratio[group]/n_new for group in split_ratio}
    else: # nothing changes if no base split
        i_new,i_base = i_good,[]

    # Iniatialisation 
    d_pid = defaultdict(list) # {PID:[index, ...], ...}
    for i in i_new:
        d_pid[df['PID'][i]].append(i)
    l_pid = [pid for pid in d_pid] # [PID, ...]
    l_ratio = list(split_ratio)
    ok, it, retry = False, 0, 0
    answers = [list(np.unique(df[item][i_new+i_base])) for item in split_sets['items']]
    tol_diff, tol_ratio = split_sets['tol_diff'], split_sets['tol_ratio']

    # Compute index range in split_index for each set, to select them in l_pid 
    start, n = 0, len(l_pid)
    split_index = ['' for k in range(len(split_ratio))]
    for k,group in enumerate(split_ratio):
        end = min(start+int(n*split_ratio[group])+1, n)
        start, split_index[k] = end, (start, end)

    # Try iteratively to find split
    while not ok and it<split_sets['max_iterations']: # 'max_iterations' x 'max_retry' at most

        # Set randomly sets by suffling l_pid and select aim proportion with split_index
        random.shuffle(l_pid)
        for k, group in enumerate(split_ratio):
            for j in range(split_index[k][0], split_index[k][1]):
                for i in d_pid[l_pid[j]]:
                    new[i] = group
        df[name] = new 

        # Count eyes for all outcomes and total for each item/set in l_cat
        l_cat = [[[0 for _ in answer_i] for answer_i in answers] for _ in range(len(split_ratio)+1)]
        for i in i_good:
            for j,item in enumerate(split_sets['items']):
                i_j = answers[j].index(df[item][i])
                l_cat[l_ratio.index(df[name][i])][j][i_j] += 1
                l_cat[-1][j][i_j] += 1

        # Compute maximum error for each outcome/item/set in l_cat
        # diff: |selected - expected|, ratio: |selected_ratio- expected_ratio|
        max_diff = max_ratio = 0
        for k, group in enumerate(_ratios):
            for i in range(len(split_sets['items'])):
                for j in range(len(l_cat[k][i])):
                    diff = abs(l_cat[k][i][j]-l_cat[-1][i][j]*_ratios[group])
                    diff_ratio = abs(l_cat[k][i][j]/l_cat[-1][i][j]-_ratios[group])
                    max_diff = max(max_diff, diff)
                    if diff_ratio>max_ratio: # save worst case to print at the end
                        worst = (round(l_cat[k][i][j]/l_cat[-1][i][j],3), _ratios[group],
                                 group, split_sets['items'][i], answers[i][j])
                    max_ratio = max(max_ratio, diff_ratio)

        # Check if error is ok and retry with easier tolerance at the end of the loop
        ok = max_diff<tol_diff or max_ratio<tol_ratio
        it += 1
        if it==split_sets['max_iterations'] and retry<split_sets['retry']: 
            tol_diff *= 1+split_sets['percent_retry']/100
            tol_ratio *= 1+split_sets['percent_retry']/100
            retry += 1
            it = 0 # reset it for new tolerance

    # Save annotation table with new split if found 
    if ok:
        df.to_csv(csv_path)
        for set_name in _ratios:
            print(f'{set_name}: {len(df[df[name]==set_name])}', end=', ')
        print()
        print(f'split found and written to table, new: {len(i_new)}, total: {len(i_good)}')
        print(f'worst imbalance: {worst}')
    else:
        print('split not found')