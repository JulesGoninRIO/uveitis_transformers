import os
import json
from itertools import product
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from image_trainer import Image_Trainer, Collage_Trainer, CNN_Trainer, Parallel_Trainer
from video_trainer import Video_Trainer
from load_dataset import load_data

import sys
sys.path.insert(1, os.path.dirname(__file__)+'/../Data/Metrics')
from Stats import plot_result_table


def main(test_mode=False):

    data, configs = load_data(test_mode=test_mode)
    if test_mode:
        configs['metrics'] = {metric: 1 for metric in configs['metrics']}
        configs['epochs'] = configs['test_mode']['epochs']
    for set_name in data:
        print(f'{set_name}:', len(data[set_name]))

    if 'video' in configs['get_name']:
        trainer = Video_Trainer(data, configs)
    elif configs['model'] == 'CNN_3D':
        trainer = CNN_Trainer(data, configs)
    elif configs['get_name'] == '4-views_col':
        trainer = Collage_Trainer(data, configs)
    elif configs['get_name'] == '4-views_par':
        trainer = Parallel_Trainer(data, configs)
    else:
        trainer = Image_Trainer(data, configs)

    trainer.run()
    return trainer.best_metrics

if __name__ == "__main__":
    main(test_mode=True)


# =====================================================

config_path = os.path.dirname(__file__)+'/configs_DL.json'
folder = os.path.dirname(__file__)+'/results/json'
_items = ['optic disk hyperfluorescence', 'macular edema', 'vascular leakage', 'capillary leakage']

def run_all(name, parameters, experiment):

    parameter_names, parameter_list = list(parameters.keys()), list(parameters.values())
    metrics = experiment['metrics']

    # Initialise results
    results = {}
    for combination in product(*parameter_list):
        current_dict = results
        for param in combination:
            if param not in current_dict:
                current_dict[param] = {}
            current_dict = current_dict[param]
        for metric in metrics:
            current_dict[metric] = [0, 0, 0, 0] if combination[0].startswith('everything') else 0
    

    # Initialise best results
    best_results = {}
    for item in results:
        [item, _] = item.split('*') if '*' in item else [item, '']
        if not item.startswith('everything'):
            best_results[item] = {metric: 0 for metric in metrics} # all for config

    # Load done experiments
    results_path = f"{folder}/results_{name}.json"
    if os.path.exists(results_path):
        results = merge_dicts(results, json.load(open(results_path)))
    best_path = f"{folder}/best_results_{name}.json"
    if os.path.exists(best_path):
        best_results = merge_dicts(best_results, json.load(open(best_path))) 

    # Run and save results
    configs = json.load(open(config_path))
    for combination in product(*parameter_list):
        print(name, combination)
        current_dict = results
        (item, model, get_name, loss, lr) = combination
        for i, param in enumerate((item, model, get_name, loss, lr)):
            current_dict = current_dict[param]
            configs[parameter_names[i]] = param # update configs
        configs['split_name'] = experiment['split']
        configs['test_set'], configs['validation_set'] = experiment['test_set'], experiment['val_set']
        if '*' in item:
            [item, bonus] = item.split('*')
            get_name += f'*{bonus}'
        if current_dict[metrics[0]] in [0, [0, 0, 0, 0]]: # train if not done
            configs['metric_select'] = metrics[0]
            if item != "everything":
                configs['metrics'] = {metric: best_results[item][metric] for metric in metrics} 
            with open(config_path, 'w') as file: # save configs before training
                json.dump(configs, file, indent=4)
            _results = main(test_mode=experiment['test_mode']) # train model
            for metric in _results:
                current_dict[metric] = _results[metric]
        if item.startswith('everything'):
            _any = ' (any)' if item=='everything (any)' else ''
            for k, item in enumerate(_items):
                item += _any
                if current_dict[metrics[0]][k] > best_results[item][metrics[0]]: # select best score
                    for metric in metrics:
                        best_results[item][metric] = current_dict[metric][k]
                    for i, x in enumerate([f'{model} {item}', get_name, loss, lr]):
                        best_results[item][parameter_names[i+1]] = x
        elif current_dict[metrics[0]] > best_results[item][metrics[0]]: # select best score
            for metric in metrics:
                best_results[item][metric] = current_dict[metric]
            for i, x in enumerate([model, get_name, loss, lr]):
                best_results[item][parameter_names[i+1]] = x
    
        with open(results_path, 'w') as file: # in the loop so saved if give up or crash
            json.dump(results, file, indent=4)
        with open(best_path, 'w') as file:
            json.dump(best_results, file, indent=4)

    df = pd.DataFrame.from_dict(best_results, orient='index')
    df.columns = df.columns.str.replace('_test', '')
    plot_result_table(df[['1-OCI', 'accuracy', 'f1-weighted', 'kappa']], 'predictions '+name)

def merge_dicts(d1, d2):
    d = d1.copy()
    for key, value in d2.items():
        if key in d and isinstance(value, dict):
            d[key] = merge_dicts(d[key], value) 
        elif value:
            d[key] = value
    return d
