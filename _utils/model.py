
import sys
import torch

import torch.nn as nn


model_shorthands = {
    'resnet18': 'r18',
    'resnet50': 'r50',
    'densenet161': 'dense161',
    'vgg16_bn': 'vgg16_bn',
    "wide_resnet50_2": "wr50_2",
}

adv_trained_models = {
    'resnet18': 'resnet18_l2_eps3.pt',
    'resnet50': 'resnet50_l2_eps3.pt',
    'densenet161': 'densenet_l2_eps3.pt',
    'vgg16_bn': 'vgg16_bn_l2_eps3.pt',
    "wide_resnet50_2": "wide_resnet50_2_l2_eps3.pt"
}

def clean_weights(weights_path, device):
    weights = torch.load(weights_path, map_location=device)
    all_w = [w for w in weights['model']]
    state_dict = weights['model']
    for w in all_w:
        wt = state_dict.pop(w)
        if w.startswith("module.attacker.model."):
            state_dict[w[22:]] = wt
    return state_dict

def get_model(arch, dataset='imagenet', train_mode='standard',):

    assert arch in ['resnet18', 'resnet50', 'densenet161', 'vgg16_bn', 'wide_resnet50_2'], "Model not supported"
    assert dataset in ['imagenet'], "Dataset not supported"
    assert train_mode in ['standard', 'adv_trained'], "Training mode not supported"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    if arch == 'resnet18':
        print("Loading ResNet18 Model")
        from torchvision.models import resnet18, ResNet18_Weights
        if train_mode == 'standard':
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            print("Standard ResNet18 Loaded Successfully")
          
        elif train_mode == 'adv_trained':
            # print working directory
            weights_path = f"./{adv_trained_models['resnet18']}"
            state_dict = clean_weights(weights_path, device)
            model = resnet18(weights="DEFAULT")
            model.load_state_dict(state_dict)
            print("Adversarially Trained ResNet18 Loaded Successfully")

    elif arch == 'resnet50':
        print("Loading ResNet50 Model")
        from torchvision.models import resnet50, ResNet50_Weights
        if train_mode == 'standard':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            print("Standard ResNet50 Loaded Successfully")
          
        elif train_mode == 'adv_trained':
            weights_path = f"./{adv_trained_models['resnet50']}"
            state_dict = clean_weights(weights_path, device)
            model = resnet50(weights="DEFAULT")
            model.load_state_dict(state_dict)
            print("Adversarially Trained ResNet50 Loaded Successfully")

    elif arch == 'densenet161':
        print("Loading DenseNet161 Model")
        from torchvision.models import densenet161, DenseNet161_Weights
        if train_mode == 'standard':
            model = densenet161(weights=DenseNet161_Weights.DEFAULT)
            print("Standard DenseNet161 Loaded Successfully")
          
        elif train_mode == 'adv_trained':
            raise NotImplementedError("Adversarial Training not supported for DenseNet161")

    elif arch == 'vgg16_bn':
        print("Loading VGG16_BN Model")
        from torchvision.models import vgg16_bn, VGG16_BN_Weights
        if train_mode == 'standard':
            model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
            print("Standard VGG16_BN Loaded Successfully")
          
        elif train_mode == 'adv_trained':
            raise NotImplementedError("Adversarial Training not supported for VGG16_BN")
    
    elif arch == 'wide_resnet50_2':
        print("Loading Wide ResNet50_2 Model")
        from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
        if train_mode == 'standard':
            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
            print("Standard Wide ResNet50_2 Loaded Successfully")
          
        elif train_mode == 'adv_trained':
            raise NotImplementedError("Adversarial Training not supported for Wide ResNet50_2")

    return model