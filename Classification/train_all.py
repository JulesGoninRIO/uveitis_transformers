from main import run_all
from gradcam import plot_saliency
import argparse

# =====================================================
# Parameters available (not all combinations)

_items = ['optic disk hyperfluorescence', 'macular edema', 'vascular leakage', 'capillary leakage']
binary_items = [item + ' (any)' for item in _items]
peripheral_items = ['vascular leakage (periphery 2)', 'capillary leakage (score 4)']
all_items = _items+peripheral_items

models = ['swin_timm', 'SWIN', 'ViT', '768_SWIN', 'DenseNet'] 

get_names = ['last_55', 'last_102', 'last_55_2']

lrs = ['2e-05', '1e-05', '5e-06']

losses = ['co', 'co', 'ce', 'occ']
weighted_losses = ['w_'+loss for loss in losses]

metrics = ['1-OCI_validation', '1-OCI_test', 'accuracy_test', 'f1-binary_test', 'f1-weighted_test', 'kappa_test']
metrics_binary = ['f1-score_validation', 'thresh', 'f1-score_test', 'auc_test']


# =====================================================
# Experiments

parameters = {'item': _items, 'model': ['swin_timm'], 'get_name': ['last_55'], 'loss': ['w_co'], 'lr': lrs}
experiment = {'split': 'Shalini', 'metrics': metrics, 'test_mode':False, 'test_set': 'test', 'val_set': 'val'}

experiments_available = {
    'default': {},
    'binary': {'item': binary_items, 'loss': ['bce'], 'metrics': metrics_binary},
    'test_mode': {'item': all_items, 'test_mode':True}
}

parser = argparse.ArgumentParser(description="Uveitis Grading DL experiments")
parser.add_argument('-exp', type=str, help='default, binary or test_mode', required=True)
name = parser.parse_args().exp
new_parameters = experiments_available[name]
parameters.update(new_parameters)
experiment.update(new_parameters)

run_all(name, parameters, experiment)

# =====================================================
# Compute saliency maps
  
print('Plotting saliency map ...')
for item in _items:
    plot_saliency('swin_timm', item, "GradCAM" , 'Shalini', keys='all')