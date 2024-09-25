from transformers_224 import SWIN, ViT
import torchvision.models as torch_models
import torch
import timm
# from transformers import VivitImageProcessor, VivitForVideoClassification

transformers_infos = {
    "script":{
        "SWIN": SWIN.SwinForImageClassification.from_pretrained,
         "ViT": ViT.ViTForImageClassification.from_pretrained},
    "model":{
        "SWIN": "microsoft/swin-base-patch4-window7-224-in22k",
        "ViT": "google/vit-base-patch16-224-in21k"
    }
}

# 384x384 input and architecture, not fixed classification layer             
def get_swin_timm():
    
    model = timm.models.swin_large_patch4_window12_384(pretrained=True)
    torch.save(model.state_dict(), 'models/swin_timm.pth')

# 224x224 architecture, fixed classification layer 
def get_transform(model_name, input_size):
        
        model_args = {"ignore_mismatched_sizes": True,
                      "pretrained_model_name_or_path": transformers_infos["model"][model_name],
                      "image_size": input_size}
        
        for n_out in [1]+[i for i in range(3, 11)]:
        
            model_args["num_labels"] = n_out
            model = transformers_infos["script"][model_name](**model_args)

            folder = 'transformers_224/'+'768_'*(input_size==768)+model_name+'_'+str(n_out)
            model.config.save_pretrained(folder)
            model.save_pretrained(folder)
             

# 768x768 input, fixed classification layer
def get_densenet():

    for n_out in range(3, 11):
            
        pretrained_densenet = torch_models.densenet121(pretrained=True)
        num_features = pretrained_densenet.classifier.in_features
        pretrained_densenet.classifier = torch.nn.Linear(num_features, n_out)
        torch.save(pretrained_densenet.state_dict(), f'DenseNet/model_{n_out}.pth')

# =====================================
# Run to save pretrained models

# get_densenet()

# get_transform('SWIN', 224)
# get_transform('ViT', 224)
# get_transform('SWIN', 768)