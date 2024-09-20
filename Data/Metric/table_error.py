import pandas as pd
import json
import warnings
import os
from shutil import copy

import sys
sys.path.insert(1, os.path.dirname(__file__)+'/../Pipeline')
from Frames import get_last

test_set = 'CV'

folder_path = os.path.dirname(__file__)+'/..'
TT_labels = json.load(open(folder_path+'/TT_labels.json'))
error_folder = folder_path+'/../Classification/results/error_table'
annotation_path = folder_path+'/Annotations/table.csv'

data_folder = 'T:/Studies/Vasculitis/data/_FA_data'
image_error_folder = data_folder+'/largest errors test set'

def json_error(item, paths, test_set):

    error_path = f"{error_folder}/{test_set}.json"
    if os.path.exists(error_path):
        errors = json.load(open(error_path))
    else:
        errors = {}
    for key in list(errors.keys()):
        if item in errors[key]:
            del errors[key][item]
        if not errors[key]:
            del errors[key]
            
    for (path, label, pred) in paths:
        label, pred = TT_labels[item][label], TT_labels[item][pred]
        if type(path) == type([]):
            path = path[0]
        folders = path.split('/')
        key = folders[5][len('vasculitis_'):]+'_'+folders[6]
        if not key in errors:
            errors[key] = {}
        errors[key][item] = (label, pred)
    with open(error_path, 'w') as file: 
        json.dump((dict(sorted(errors.items()))), file, indent=4)

def sum_errors(test_sets):
    all_errors = {}
    for test_set in test_sets:
        path = f"{error_folder}/{test_set}.json"
        if not os.path.exists(path):
            print("no error file or sum already done")
            return
        all_errors.update(json.load(open(path)))
        os.remove(path)
    with open(f"{error_folder}/CV.json", 'w') as file: 
        json.dump((dict(sorted(all_errors.items()))), file, indent=4)

def gdoc_error(error_path):
    
    errors = json.load(open(error_path))
    annotations = pd.read_csv(annotation_path, index_col=0)
    keys = list(annotations['key'])
    
    for bias in ['label --- pred', 'label', '']:
        
        df = pd.DataFrame([], columns=['key', 'optic disk hyperfluorescence', 
                        'macular edema', 'vascular leakage', 'capillary leakage'])
        
        for key, values in errors.items():
            
            url = annotations['url'][keys.index(key)]
            row = {'key': f'=HYPERLINK("http://{url}", "{key}")'}
            for item, error in values.items():
                row[item] = error[0]*('label' in bias)+' -- '+error[1]*('pred' in bias)
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = df.append(row, ignore_index=True)
                
        with pd.ExcelWriter(f"{error_path[:-5]}__{bias}.xlsx", engine='openpyxl') as writer:
            df.to_excel(writer, index=False)

def find_image_errors(error_path):
    get_last_55 = get_last(1)
    for key in json.load(open(error_path)):
        [pid, date, side] = key.split('_')
        study = f"{data_folder}/results/vasculitis_{pid.zfill(4)}/{date}_{side}"
        copy(f"{study}/{get_last_55(study)[0]}", f"{image_error_folder}/{key}.png")
    for bias in ['label --- pred', 'label', '']:
        if os.path.exists(error_path[:-5]+bias+'.xlsx'):
            copy(f"{error_path[:-5]}__{bias}.xlsx", f"{image_error_folder}/label -- prediction.xlsx")
        
        
error_path = f"{error_folder}/{test_set}.json"
if __name__ == '__main__':
    if test_set == 'CV':
        sum_errors(['A', 'B', 'C', 'D'])
    gdoc_error(error_path)
    if os.path.exists(data_folder):
        find_image_errors(error_path)