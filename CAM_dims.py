import numpy as np
import torch
import tqdm
import os
import sys
import torch.hub
import os
from torchvision import transforms, datasets
import argparse
import pickle

def load_state_dict_from_url(*args, **kwargs):
    return torch.hub.load_state_dict_from_url(*args, **kwargs)

# Create a dummy module
class DummyModule:
    def __init__(self):
        self.load_state_dict_from_url = load_state_dict_from_url

# Replace the faulty import
sys.modules['torchvision.models.utils'] = DummyModule()

from robustness.datasets import CIFAR
from robustness import model_utils

def get_activations(model, dl):
  activations = []
  # hook to get input features of a layer 
  def activation_hook(module, input, output):
    in_feats = input[0].detach().cpu().numpy()
    activations.append(in_feats)
    return

  # register hook at the final classification layer (input of final layer == activations of penultimate/representation layer)
  h1 = model.linear.register_forward_hook(activation_hook)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model.eval()
  with torch.no_grad():
    for images, labels, in tqdm.tqdm(dl):
      images = images.to(device)
      labels = labels.to(device)
      _ = model(images)
  h1.remove()

  # remove last batch if it has less than BS points
  if len(activations[-1]) != len(activations[0]):
    activations = activations[:-1]

  # reshape to remove batches
  activations_arr = np.array(activations)
  NB, BS, A = activations_arr.shape
  activations_r = activations_arr.reshape(NB*BS, A)
  
  print("Activations Shape: ",activations_r.shape)
  return activations_r

def main():

  parser = argparse.ArgumentParser(description='Get Class Activations')
  parser.add_argument('--model_type', type=str, help='Type of model: standard, adv_trained or robust', default='standard')
  parser.add_argument('--data_split', type=str, help='Which data loader to use: train or test', default='test')
  args = parser.parse_args()

  model_ext = ''
  if args.model_type == 'adv_trained':
    model_ext = '_adv'
  elif args.model_type == 'robust':
    model_ext = '_robust'

  model_path = f'./cifar_r50{model_ext}_train/checkpoint.pt.latest'

  ds = CIFAR('./data')
  print("Trying to load model from path: ", model_path)
  model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path='cifar_nat.pt')

  train_loader , test_loader = ds.make_loaders(batch_size=1, workers=1)
  dl = None
  if args.data_split == 'train':
    dl = train_loader
  elif args.data_split == 'test':
    dl = test_loader
  

  class_activations = {i: [] for i in range(10)}

  print("Getting Class Activations ...")
  for input, label in tqdm(dl):
    input, label = input.cuda(), label.cuda()
    # get class index
    label = label.item()

    activations = get_activations(model, input)

    # append to class activations
    class_activations[label].append(activations)

  print("Class Activations obtained.")
  for key in class_activations:
    class_activations[key] = np.array(class_activations[key])
    print(f"Class {key} Activations Shape: ", class_activations[key].shape)


  print("Saving Class Activations ...") 
  with open(f'./cifar_r50{model_ext}_train/class_acts_{args.data_split}.pkl', 'wb') as f:
    pickle.dump(class_activations)
  
  return

# TODO: Test the code (and Manifold Dim Est)

if __name__ == '__main__':
  main()





