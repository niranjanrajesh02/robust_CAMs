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
import pickle
import os
import numpy as np
from _utils.model import get_model, model_shorthands
from _utils.attacks import prepare_attack
from _utils.data import get_dataloader

def get_classwise_acc(model, attack, eps, test_loader, num_classes=1000, device=None, model_type='standard'):
  class_correct = {i: 0 for i in range(num_classes)}
  class_total = {i: 0 for i in range(num_classes)}

  print("Getting Classwise Accuracy for epsilon: ", eps)

  for inputs, labels in tqdm(test_loader):
    if eps != 0: #* Adversarial Evaluation
      inputs = inputs.detach().cpu().numpy().astype(np.float32)
      labels = labels.detach().cpu().numpy().astype(np.float32)
      adv_input,_,_ = attack(model, inputs, labels, epsilons=[eps])
      adv_input = adv_input[0]
      output = model.predict(adv_input)
      # print(adv_input.shape, output.shape)
      # print(labels.shape)
      preds = np.argmax(output, axis=1)

    else: #* Standard Evaluation
      # inputs = inputs.detach().cpu().numpy().astype(np.float32)
      # labels = labels.detach().cpu().numpy().astype(np.float32)
      output = model(inputs)
      preds = np.argmax(output.detach(), axis=1)

    # classwise accuracy calculation
    for i in range(len(labels)):
        label = int(labels[i])
        pred = int(preds[i])
        # print(label, pred, int(pred == label))
        class_correct[label] += int(pred == label)
        class_total[label] += 1
  
  classwise_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)}
  print("Classwise Accuracy Calculated Successfully")
  return classwise_acc


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_arch', type=str, help='Name of the model to evaluate', default='resnet50')
  parser.add_argument('--model_type', type=str, help='Type of model: standard, adv_trained', default='standard')
  parser.add_argument('--adv_evaluate', type=bool, help='Adversarially Evaluate trained model', default=False)
  parser.add_argument('--l_constraint', type=str, help='Type of constraint: l2, linf', default=None)
  parser.add_argument('--dataset', type=str, help='Dataset to use (cifar, restricted_imagenet, imagenet)', default='imagenet')
  args = parser.parse_args()
  args.dataset = args.dataset.lower()

  assert args.dataset in ['imagenet'], "Invalid dataset" 
  assert args.model_type in ['standard', 'adv_trained'], "Invalid model type"
  assert args.model_arch in ['resnet18', 'resnet50', 'densenet161', 'vgg16_bn', 'wide_resnet50_2'], "Model not supported"
  assert args.l_constraint in ['l2', 'linf', None], "Invalid constraint type"
  
  if args.adv_evaluate:
    assert args.l_constraint is not None, "Constraint type not provided for adversarial evaluation"

  print("\n\n=============================================")
  print(f"Dataset: {args.dataset}, Model: {args.model_arch}, Model Type: {args.model_type}, L_Constraint: {args.l_constraint if args.adv_evaluate else 'None'}")
  print("=============================================")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device: ", device)

  # * Loading the Model
  model_ext = ''
  model = None
  model_short =  model_shorthands[args.model_arch]

  if args.model_type == 'adv_trained':
    model_ext = f'{model_short}_adv'
    model = get_model(arch=args.model_arch, dataset=args.dataset, train_mode='adv_trained')
    model = model.to(device)

  elif args.model_type == 'standard':
    model_ext = f'{model_short}'
    model = get_model(arch=args.model_arch, dataset=args.dataset, train_mode='standard')
    model = model.to(device)
  assert model is not None, "Model not loaded successfully"
  model.eval()

  #* Loading the dataset
  val_loader = None
  if args.dataset == 'imagenet':
    val_loader = get_dataloader(ds_name='imagenet', split='val', bs=32)
  
  # * Prepare the attack
  attack_params = None
  if args.l_constraint == 'l2':
    attack_params = {'attack_type': 'L2_PGD', 'epsilon': 3, 'iterations': 7}
    fmodel, attack = prepare_attack(model, attack_params)
  elif args.l_constraint == 'linf':
    attack_params = {'attack_type': 'Linf_PGD', 'epsilon': 8/255, 'iterations': 7}
    fmodel, attack = prepare_attack(model, attack_params)
  else:
    attack_params = {'attack_type': 'None', 'epsilon': 0, 'iterations': 0}
    fmodel = model
    attack = None

  print("Model and Attack Prepared with params: ", attack_params)


  class_accuracies = get_classwise_acc(fmodel, attack, attack_params['epsilon'], val_loader, num_classes=1000, device=device, model_type=args.model_type)
  
  save_path= f'./{args.dataset}_{model_ext}_train'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  # print("Classwise Accuracies: ", class_accuracies)
  with open(f'./{save_path}/classwise_acc_{args.l_constraint}.pkl', 'wb') as f:
    pickle.dump(class_accuracies, f)

  print("Classwise Accuracies saved successfully to ", save_path)
  return

if __name__ == '__main__':
  main()