import json
import os
import pandas as pd

import sys
git_folder = os.path.dirname(__file__)+'/..'
sys.path.insert(1, git_folder+'/Data/Pipeline')
from Frames import get_last, get_4_views

folder = "T:/Studies/Uveitis/data/_FA_data" if os.path.exists('T:/Studies') else "/data/soin/vasculitis"
annotations = pd.read_csv(git_folder+'/Data/Annotations/table.csv', index_col=0)
TT_labels = json.load(open(git_folder+'/Data/TT_labels.json'))
pids, keys = list(annotations['PID']), list(annotations['key'])

patients = [patient for patient in os.listdir(folder+'/results') if int(patient[11:]) in pids]
main_items = ['optic disk hyperfluorescence', 'macular edema', 'vascular leakage', 'capillary leakage']


# Initialase and create data json
def load_data(new_configs={}, test_mode=False):

    configs = json.load(open(git_folder+'/Classification/configs_DL.json'))
    configs.update(new_configs)
    item, get_name = configs['item'], configs['get_name']
    n_images = 1

    # Select get_path from get_name
    if get_name.startswith('last_') or get_name.startswith('all_'):
        [times, angle] = get_name.split('_')
        [angle, n_images] = angle.split('*') if '*' in angle else [angle, 1]
        n_images = int(n_images)
        get_path = get_last(angle=angle, n=n_images, times=times)
    else:
        raise ValueError(f"get_name input {get_name} not implemented")

    items = main_items if item=='everything' else [item]
    items = [item+' (any)' for item in main_items] if item=='everything (any)' else [item]
    data = {'train':[], 'test':[], 'validation':[]}
    set_rename = {configs['test_set']: 'test', configs['validation_set']: 'validation', 'val': 'validation'}

    # Iterate and get annotated studies
    for patient in patients:
        for study in os.listdir(folder+'/results/'+patient):
            pid = str(int(patient[11:]))
            key = pid+'_'+study
            if key in keys:

                images = get_path(folder+'/results/'+patient+'/'+study)

                if images:
                    
                    if os.path.exists('T:/Studies'):
                        pid = pid.zfill(4)
                    paths = [f"{folder}/results/vasculitis_{pid}/{study}/{image}" for image in images]
                    i = keys.index(key)
                    set_name = annotations[configs['split_name']][i]

                    # Assign single letter set_name to set based on configs
                    set_name = set_rename[set_name] if set_name in set_rename else set_name
                    set_name = 'train' if len(set_name)== 1 else set_name

                    # Get label
                    new_label, ok = [], True
                    for item in items:
                        label = annotations[item][i]
                        if label not in ['not assessable', 'na']:
                            if item.endswith('(any)') or item.endswith('(significant)'): # Binary label
                                new_label.append(int(label in ['any', 'significant']))
                            else: # Multiclass label
                                new_label.extend([int(str(label)==cat) for cat in TT_labels[item]])
                        else:
                            ok = False
                            
                    # Fill splits with selected [path, label]
                    if ok and set_name in data: 
                        configs['n_out'] = len(new_label)
                        new_label = new_label[0] if configs['n_out']==1 else new_label
                        if n_images > 1:
                            if len(paths) == n_images:
                                data[set_name].append([paths[:], new_label] )
                        else:
                            data[set_name].extend([[path, new_label] for path in paths])

    if test_mode:
        for set_name in data:
            data[set_name] = data[set_name][:int(len(data[set_name])*configs['test_mode']['data_%_used']/100)]
    if not data['train']:
        raise ValueError(f"no image found for training (get_name: {get_name}, model: {configs['model']})")

    # Data Augmentation
    if configs['data_augmentation']['miror']:
        data['train'].extend([['m*'+path, label] for [path, label] in data['train']])
    if configs['data_augmentation']['rotate']:
        data['train'].extend([['r1*'+path, label] for [path, label] in data['train']]
                              +[['r2*'+path, label] for [path, label] in data['train']])

    return data, configs