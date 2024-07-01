import sys
import torch
def load_state_dict_from_url(*args, **kwargs):
    return torch.hub.load_state_dict_from_url(*args, **kwargs)

# Create a dummy module
class DummyModule:
    def __init__(self):
        self.load_state_dict_from_url = load_state_dict_from_url
sys.modules['torchvision.models.utils'] = DummyModule()


import sys
import torch

sys.path.append('..')
import vonenet
sys.path.append('../_utils')

from torchvision.models import resnet50

def get_model(arch, dataset='imagenet', train_mode='standard', weights_path=None, with_transforms=False):

    assert arch in ['resnet50', 'vone_resnet'], "Model not supported"
    assert dataset in ['imagenet'], "Dataset not supported"
    assert train_mode in ['standard', 'adv_trained'], "Training mode not supported"
    if train_mode != 'vone_resnet':
        assert weights_path is not None, "Weights path not provided"
    
    transforms = None
    if arch == 'resnet50':
        print("Loading ResNet50 Model")
        if train_mode == 'standard':
            model = resnet50(weights=None)
            model.load_state_dict(torch.load(weights_path))
            print("Standard ResNet50 Loaded Successfully")
            if with_transforms:
                from torchvision.models import ResNet50_Weights
                transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()
        elif train_mode == 'adv_trained':
            assert torch.cuda.is_available() == True, "Adversarial training is only supported on GPU"
            from robustness import model_utils
            from robustness.datasets import ImageNet
            ds = ImageNet('')
            model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=weights_path)
            model = model.model
            print("Adversarially Trained ResNet50 Loaded Successfully")
        

    elif arch == 'vone_resnet':
        assert train_mode == 'standard', "VOneNet model is not available in adv_trained mode"
        print("Loading VOneNet Model")
        model = vonenet.get_model(model_arch='resnet50', pretrained=True, noise_mode=None)
        print("VOneNet Loaded Successfully")

    if with_transforms:
        return model, transforms
    else:
        return model