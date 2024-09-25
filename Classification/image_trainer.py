from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.models as torch_models
import timm
from transformers import AutoModelForImageClassification
import torch
import numpy as np

import os
import sys
sys.path.insert(1, os.path.dirname(__file__)+'/..')
from trainer import Trainer

models_folder =  os.path.dirname(__file__)+'/models'


class Image_Trainer(Trainer):

    def __init__(self, data, configs):

        super().__init__(data, configs)

        self.save_torch = self.configs["model"] in ['swin_timm', 'DenseNet']
        self.multi = '*' in self.configs['get_name'] or self.configs['get_name']=='4-views_sep'
        self.size = 768 if self.configs["model"] in ['768_SWIN', 'DenseNet'] else 384
        self.n_frames = 1 if type(data['train'][0][0])==type('') else len(data['train'][0][0])

    def swin_timm(self):

        model = timm.models.swin_large_patch4_window12_384(pretrained=False)
        model.load_state_dict(torch.load(f'{models_folder}/Swin_timm/{self.configs["model"]}.pth'))
        model.head = torch.nn.Linear(model.head.in_features, self.n_out)

        return model
    
    def load_image(self, path):

        gray_image = TF.to_grayscale(Image.open(path), num_output_channels=3)
        image = TF.to_tensor(gray_image).unsqueeze(0).to(self.device)
        return TF.resize(image, (self.size, self.size))

    def load_model(self):

        if self.configs["model"] == 'DenseNet':
            self.model = torch_models.densenet121(pretrained=False)
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, self.n_out)
            self.model.load_state_dict(torch.load(models_folder+'/DenseNet/Densenet_'+str(self.n_out)+'model.pth'))
        elif self.configs["model"] == 'swin_timm':
            self.model = self.swin_timm()
        else:
            folder = self.configs['SOIN_save']+'/pretrained_models/'+self.configs["model"]+'_'+str(self.n_out)
            self.model = AutoModelForImageClassification.from_pretrained(folder) 

    def apply_model(self, path):

        # Open image
        path_ = path.split('-')[1] if '*' in path else path
        input = self.load_image(path_)
        if path.split('/')[-3][-1] == 'L':
            input = TF.hflip(input)

        # Data augmentation
        if path.startswith('m'):
            input = TF.hflip(input)
        if path.startswith('r'):
            input = TF.rotate(input, 10 if path[1]=='1' else -10)

        # Apply model
        output = self.model(input)
        if not self.save_torch:
            output = output.logits
        return output.view(1, self.n_out)
    

class Parallel_Trainer(Image_Trainer):

    def __init__(self, data, configs):

        super().__init__(data, configs)

    def load_model(self):

        if self.configs["model"] != 'swin_timm':
            raise ValueError(f"Only implemented with swin_timm model")
        
        self.model = torch.nn.ModuleList()

        for _ in range(self.n_frames):
            model = self.swin_timm()
            features = model.head.in_features
            model.head = torch.nn.Identity()  # Remove the head
            self.model.append(model)

        self.model.append(torch.nn.Linear(self.n_frames*features, self.n_out))

    def apply_model(self, paths):

        outputs = [self.model[i](self.load_image(paths[i])) for i in range(self.n_frames)]
        return self.model[-1](torch.cat(outputs, dim=1))
    
    
class Collage_Trainer(Image_Trainer):

    def __init__(self, data, configs):

        super().__init__(data, configs)

        self.size = self.size//2

    def apply_model(self, paths):

        # Open image and create collage
        images = [self.load_image(paths[i]) for i in range(self.n_frames)]
        merged_input = torch.zeros(1, 3, 2*self.size, 2*self.size).to(self.device)
        merged_input[:, :, :self.size, :self.size] = images[0]  # Top-left
        merged_input[:, :, :self.size, self.size:] = images[1]  # Top-right
        merged_input[:, :, self.size:, self.size:] = images[2]  # Bottom-right
        merged_input[:, :, self.size:, :self.size] = images[3]  # Bottom-left

        # Apply model
        output = self.model(merged_input)
        if not self.save_torch:
            output = output.logits
        return output.view(1, self.n_out)
    

class CNN_Trainer(Trainer):

    def __init__(self, data, configs):

        super().__init__(data, configs)

        self.multi = False
        self.save_torch = True

    def load_model(self):

        self.model = torch_models.video.r3d_18(pretrained=False)
        self.model.load_state_dict(torch.load(models_folder+'/Vivit/CNN_3D.pth'))
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.n_out)

    def apply_model(self, paths):

        images = np.zeros((1, 3, self.n_frames, 768, 768), dtype=np.uint8)

        for i, img_path in enumerate(paths):
            img = np.array(Image.open(img_path).convert('L')) 
            img_rgb = np.transpose(np.repeat(img[:, :, np.newaxis], 3, axis=2), (2, 0, 1))
            images[0, :, i, :, :] = img_rgb

        return self.model(torch.tensor(images, dtype=torch.float32).to(self.device))