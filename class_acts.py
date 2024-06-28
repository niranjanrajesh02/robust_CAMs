import numpy as np
import torch
from tqdm import tqdm
import os
import sys
import torch.hub
import os
import argparse
import pickle
from _utils import twoNN



def load_state_dict_from_url(*args, **kwargs):
    return torch.hub.load_state_dict_from_url(*args, **kwargs)

# Create a dummy module
class DummyModule:
    def __init__(self):
        self.load_state_dict_from_url = load_state_dict_from_url

# Replace the faulty import
sys.modules['torchvision.models.utils'] = DummyModule()



def get_activations(model, input):
  activations = []
  # hook to get input features of a layer 
  def activation_hook(module, input, output):
    in_feats = input[0].detach().cpu().numpy()
    activations.append(in_feats)
    return
  
 
  h1 = None 
  # register hook at the final classification layer (input of final layer == activations of penultimate/representation layer)
  for name, module in model.named_children():
    if name == 'linear':
      h1 = module.register_forward_hook(activation_hook)
    elif name == 'fc':
      h1 = module.register_forward_hook(activation_hook)

  
  if h1 == None:
    print("No Linear Layer found in the model.")
    return
  
  model.eval()
  for param in model.parameters():
    param.requires_grad = False
   
  _ = model(input) # forward pass, activations are stored by the hook

  h1.remove()

  activations_arr = np.array(activations[0][0])

  # reshape to remove batches
  # print("Activations Shape: ",activations_arr.shape)
  # NB, BS, A = activations_arr.shape
  # activations_r = activations_arr.reshape(NB*BS, A)
  
  return activations_arr

def estimate_manifold_dim(model_ext, dataset_name='cifar'):
  print("Estimating Manifold Dimension ...")
  class_acts_file = f'./{dataset_name}_r50{model_ext}_train/class_acts_test.pkl'
  print("Class Acts File: ", class_acts_file)
  # load class activations
  if os.path.exists(class_acts_file):
    with open(class_acts_file, 'rb') as f:
      class_activations = pickle.load(f)
  else:
    print("Class Activations File not found.")
    return
  
  class_dims = {i : 0 for i in range(10)}
  for key in class_activations:
    acts = np.array(class_activations[key])
    print(f"Estimating manifold dimension for class {key} ...")
    id, _ = twoNN.estimate_dim(acts)
    print(f"Estimated manifold dimension for class {key}: ", id)
    class_dims[key] = id

  class_dims['all'] = 0
  # concatenate all class activations
  all_acts = np.concatenate([class_activations[key] for key in class_activations], axis=0)
  print("Estimating manifold dimension for all classes with concatenated activations: ", all_acts.shape)
  id, _ = twoNN.estimate_dim(all_acts)
  print("Estimated manifold dimension for all classes: ", id)
  class_dims['all'] = id
  
  print("Classwise Estimated Manifold Dimensions: ", class_dims)

  with open(f'./{dataset_name}_{model_ext}_train/class_dims_test.pkl', 'wb') as f:
    pickle.dump(class_dims, f)

  return

def main():

  parser = argparse.ArgumentParser(description='Get Class Activations')
  parser.add_argument('--dataset', type=str, help='Dataset to use (cifar, restricted_imagenet, imagenet)', default='cifar')
  parser.add_argument('--model_type', type=str, help='Type of model: standard, adv_trained or robust', default='standard')
  parser.add_argument('--task', type=str, help='Task to perform: acts or dims', default='acts')

  args = parser.parse_args()
  args.dataset = args.dataset.lower()
  
  model_ext = ''
  if args.model_type == 'adv_trained':
    model_ext = '_adv'
  elif args.model_type == 'robust':
    model_ext = '_robust'

  if args.task == 'acts':
    from robustness.datasets import CIFAR, ImageNet, RestrictedImageNet
    from robustness import model_utils

    if args.dataset == 'cifar':
      ds = CIFAR('./data')
    elif args.dataset == 'restricted_imagenet':
      ds = RestrictedImageNet('./data/imagenet')
    elif args.dataset == 'imagenet':
      ds = ImageNet('./data/imagenet')
    print("Dataset Found. Loading Model ...")

    if args.dataset == 'cifar':
      model_path = f'./cifar_r50{model_ext}_train/checkpoint.pt.latest'
    else:
      model_path = f'./models/{args.dataset}_r50{model_ext}_train.pt'
    print("Trying to load model from path: ", model_path)

    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=model_path)

    # batch size 1 allows us to get class of each image to sort into class_activations
    dl = ds.make_loaders(batch_size=1, workers=1, only_val=True)[1]
    
    num_classes = 10
    if args.dataset == 'imagenet':
      num_classes = 1000
    elif args.dataset == 'restricted_imagenet':
      num_classes = 9

    class_activations = {i: [] for i in range(num_classes)}

    print("Getting Class Activations ...")

    for input, label in tqdm(dl):
      input, label = input.cuda(), label.cuda()
      # get class index and corresponding activations
      label = label.item()
      activations = get_activations(model.model, input)
      # append to class activations
      class_activations[label].append(activations)

    print("Class Activations obtained.")

    for key in class_activations:
      class_activations[key] = np.array(class_activations[key])
      print(f"Class {key} Activations Shape: ", class_activations[key].shape)

    print("Saving Class Activations ...") 
    
    save_path= f'./{args.dataset}_r50{model_ext}_train'
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    with open(f'{save_path}/class_acts_test.pkl', 'wb') as f:
      pickle.dump(class_activations, f)
    return

  elif args.task == 'dims':
    estimate_manifold_dim(model_ext, dataset_name=args.dataset)
    return


if __name__ == '__main__':
  main()





