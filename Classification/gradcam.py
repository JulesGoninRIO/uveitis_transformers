import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import transformers
from PIL import Image
import os
import json
import numpy as np
import timm
from tqdm import tqdm
import cv2

from load_dataset import load_data

from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, RandomCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

_methods = {"GradCAM": GradCAM,
            "GradCAM++": GradCAMPlusPlus,
            "EigenGradCAM": EigenGradCAM,
            "RandomCAM": RandomCAM}

git_folder = os.path.dirname(__file__)+'/..'
saliency_output = '/data/soin/vasculitis/transformers/saliency_maps'

image_size = 384
saved_models = '/data/soin/vasculitis/transformers/trained_models/'
layers = {'SWIN':transformers.models.swin.modeling_swin.SwinAttention,
          'ViT':transformers.models.vit.modeling_vit.ViTSelfAttention}

# ===============================================================================
# ===============================================================================
# ===============================================================================

def plot_saliency(model, item, method, split, keys='disease',
                test_set='test', get_name='last_55'):

    # Initialisation
    name = f'{test_set}/{model}__{item}'
    data, _ = load_data({'item':item, 'get_name':get_name, 'split_name':split, 'test_set':test_set})
    TT_labels = json.load(open(git_folder+'/Data/TT_labels.json'))
    n_out = len(TT_labels[item])
 
    # Load model
    model = timm.models.swin_large_patch4_window12_384(pretrained=False)
    model.head = torch.nn.Linear(model.head.in_features, n_out)
    model.load_state_dict(torch.load(saved_models+name+'.pth'))
    model.eval()

    # Select image paths
    paths = []
    for (path, label_int) in data['test']:
        label_int = np.argmax(label_int)
        folders = path.split('/')
        key = folders[-4].split('_')[-1]+folders[-3]
        if keys=='all' or (keys=='disease' and label_int) or key in keys:
            paths.append((path, key, label_int))

    # Compute saliency for each image
    scores = {}
    for (path, key, label_int) in tqdm(paths, desc=item):

        # Load image to tensor
        image = Image.open(os.path.join(path)).resize((image_size, image_size))
        if path.split('/')[-3][-1] == 'L':
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        gray_image = TF.to_grayscale(image, num_output_channels=3)
        input_tensor = TF.to_tensor(gray_image).unsqueeze(0)

        # Run model prediction
        with torch.no_grad(): 
            output = model(input_tensor)
            likelihoods = F.softmax(output, dim=1).cpu().numpy().tolist()[0]
            predicton = TT_labels[item][int(torch.argmax(output.view(1,n_out)))]

        # Compute saliency
        (score, saliency) = set_saliency(model, input_tensor, image, method, label_int)

        # Fill score results
        label = TT_labels[item][label_int]
        scores[key] = {'ROAD': float(score),'label': label, 'predicton': predicton,
                    'likelihood': likelihoods[label_int], 'likelihoods': likelihoods}
        score_prefix = str(round(score, 3))+'_'+key

        # Save saliency (2 folder structures)
        for (folder_order, prefix) in [(f'_study/{key}', item), (f'_item/{item}', score_prefix)]:
            save_folder = f'{saliency_output}{folder_order}'
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            save_path = f'{save_folder}/{prefix}__{method}__{test_set} -l {label} -p {predicton}.jpg'
            saliency.save(save_path)

    # Save score results
    with open(f'results/saliency_map/{item}.json', 'w') as json_file:
        json.dump(scores, json_file, indent=4)

# ===============================================================================
# ===============================================================================
# ===============================================================================

# Show metric on top of the CAM
def visualize_score(visualization, score, name, percentiles):
    visualization = cv2.putText(visualization, name, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "(Least first - Most first)/2", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Percentiles: {percentiles}", (10, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    visualization = cv2.putText(visualization, "Remove and Debias", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA) 
    visualization = cv2.putText(visualization, f"{score:.5f}", (10, 85), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    return visualization

# Set up and compute saliency
def set_saliency(model, input_tensor, image, method, label_int, smooth='', show_text=False):

    # Compare all methods
    if method == 'benchmark':
        return benchmark(model, input_tensor, image, label_int)
    
    # Get optional arguments
    eigen_smooth = aug_smooth = metric = False
    if '_' in method:
        [method, smooth] = method.split('_')
        eigen_smooth = ('eigen' in smooth)
        aug_smooth = ('aug' in smooth)
        metric = ('metric' in smooth)

    # Compute single saliency map
    return benchmark(model, input_tensor, image, label_int, methods=[method],
                     metric=metric, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth, show_text=show_text)

# Compute saliency
def benchmark(model, input_tensor, image, label_int, methods=_methods,
              metric=True, eigen_smooth=False, aug_smooth=False, show_text=False):

    target_layers = [model.layers[-1].blocks[-1].norm2]
    targets = [ClassifierOutputTarget(label_int)]

    # Reshape attention layer output
    def reshape_transform(tensor, height=12, width=12):
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        return result.transpose(2, 3).transpose(1, 2)

    # ROAD inialisation
    if metric:
        percentiles = [20, 40, 60, 80]
        cam_metric = ROADCombined(percentiles=percentiles)
        metric_targets = [ClassifierOutputSoftmaxTarget(label_int)]
    
    # Compute saliency for each method
    visualizations = []
    for name in methods:

        # Iniatialise CAM
        if name == "AblationCAM":
            cam = _methods[name](model=model, target_layers=target_layers, 
                              reshape_transform=reshape_transform, ablation_layer=AblationLayerVit())
        else:
            cam = _methods[name](model=model, target_layers=target_layers, 
                                reshape_transform=reshape_transform)
        
        # Compute Saliency
        attributions = cam(input_tensor=input_tensor, targets=targets, 
                           eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[0, :]
        image_np = np.array(image.convert('RGB'), dtype=np.float32)/255
        visualization = show_cam_on_image(image_np, attribution, use_rgb=True)
        
        # Add ROAD score to map
        score = 0
        if metric:
            score = cam_metric(input_tensor, attributions, metric_targets, model)[0]
            if show_text:
                visualization = visualize_score(visualization, score, name, percentiles)

        visualizations.append(visualization)
        
    return (score, Image.fromarray(np.hstack(visualizations)))