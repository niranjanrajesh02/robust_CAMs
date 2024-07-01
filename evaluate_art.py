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
import pickle
from tqdm import tqdm
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import vonenet

import pickle
import os
import numpy as np
from torchvision.models import resnet50
from _utils.model import get_model
from _utils.attacks import prepare_attack, prepare_art_attack

def get_classwise_acc(model, attack, eps, test_loader, num_classes=1000, device=None, model_type='standard'):
  class_correct = {i: 0 for i in range(num_classes)}
  class_total = {i: 0 for i in range(num_classes)}

  print("Getting Classwise Accuracy for epsilon: ", eps)

  for inputs, labels in tqdm(test_loader):
    
    if model_type != 'vone_resnet':
      if eps != 0:
        img_adv, _, _ = attack(model, inputs, labels, epsilons=[eps])
        # Generate adversarial examples
        img_adv = img_adv[0]
        outputs = model(img_adv)
        preds = torch.argmax(outputs, dim=1)
      
      else:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
      
    else:
      inputs = inputs.detach().cpu().numpy().astype(np.float32)
      labels = labels.detach().cpu().numpy().astype(np.float32)
      adv_input = attack.generate(x=inputs, y=labels)
      output = model.predict(adv_input)
      print(adv_input.shape, output.shape)
     


    for i in range(len(labels)):
        label = labels[i].item()
        pred = preds[i].item()
        class_correct[label] += int(pred == label)
        class_total[label] += 1

  classwise_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)}

  return classwise_acc


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type', type=str, help='Type of model: standard, adv_trained, vone_resnet or robust', default='standard')
  parser.add_argument('--eps', type=float, help='Epsilon value for adversarial training', default=0)
  parser.add_argument('--dataset', type=str, help='Dataset to use (cifar, restricted_imagenet, imagenet)', default='imagenet')
  args = parser.parse_args()
  args.dataset = args.dataset.lower()

  assert args.dataset in ['imagenet'], "Invalid dataset" 
  assert args.model_type in ['standard', 'adv_trained',  'vone_resnet'], "Invalid model type"

  print("\n\n=============================================")
  print(f"Dataset: {args.dataset}, Model Type: {args.model_type}, Epsilon: {args.eps}")
  print("=============================================")
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # device = torch.device("cpu") 
  print("Device: ", device)

  # * Loading the Model
  model_ext = ''
  model = None
  
  if args.model_type == 'adv_trained':
    model_ext = '_adv'
    model_path = f'./models/{args.dataset}_r50{model_ext}_train.pt'
    model = get_model(arch='resnet50', dataset=args.dataset, train_mode='adv_trained', weights_path=model_path).to(device)
    preproc = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
  elif args.model_type == 'standard':
    from torchvision.models import ResNet50_Weights
    # preproc = ResNet50_Weights.IMAGENET1K_V2.transforms()
    model_path = f'./models/{args.dataset}_r50{model_ext}_train.pt'
    model = get_model(arch='resnet50', dataset=args.dataset, train_mode='standard', weights_path=model_path)
    model = model.to(device)


  elif args.model_type == 'vone_resnet':
    model_ext = '_vone'
    model = get_model(arch='vone_resnet', dataset=args.dataset, train_mode='standard', weights_path=None).to(device)
    
  assert model is not None, "Model not loaded successfully"
  model.eval()
  

  val_loader = None
  #* Loading the dataset
  if args.dataset == 'imagenet':
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.ImageFolder(root='./data/imagenet/val', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)
  
  # * Prepare the attack

  attack_params = {'attack_type': 'L2_PGD', 'epsilon': args.eps, 'iterations': 7}
  if args.model_type == 'vone_resnet':
    model, attack = prepare_art_attack(model, attack_params)
    print("ART Model and Attack Prepared with params: ", attack_params)
  else:
    fmodel, attack = prepare_attack(model, attack_params, transforms=transform)
    print("Foolbox Model and Attack Prepared with params: ", attack_params)


  class_accuracies = get_classwise_acc(fmodel, attack, args.eps, val_loader, num_classes=1000, device=device, arch=args.model_type)
  
  save_path= f'./{args.dataset}_r50{model_ext}_train'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  print("Classwise Accuracies: ", class_accuracies)
  with open(f'./{save_path}/classwise_acc_e{args.eps}.pkl', 'wb') as f:
    pickle.dump(class_accuracies, f)

  print("Classwise Accuracies saved successfully to ", save_path)
  return

if __name__ == '__main__':
  main()