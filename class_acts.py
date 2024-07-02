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
from _utils.model import get_model
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_layer_accs(model, input):
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
  acts = np.array(activations)
  # print("Activations shape:", acts.shape)
  activations_arr = np.array(activations[0])
  return activations_arr

def get_activations(model, dl, device, bs=1):    
  num_classes = 1000

  class_activations = {i: [] for i in range(num_classes)}

  print("Getting Class Activations ...")

  for inputs, labels in tqdm(dl):
    inputs, labels = inputs.to(device), labels.to(device)
    # get class index and corresponding activations !
    activations = get_layer_accs(model, inputs)
    # print("Activations Shape: ", activations.shape)
    # print("Labels Shape: ", labels.shape)
    # append to class activations

    for i in range(len(labels)):
      label = labels[i].item()
      class_activations[label].append(activations[i])
    
    

  print("Class Activations obtained.")

  for key in class_activations:
    class_activations[key] = np.array(class_activations[key])
    # print(f"Class {key} Activations Shape: ", class_activations[key].shape)

  return class_activations

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
  for key in tqdm(class_activations):
    acts = np.array(class_activations[key])
    # print(f"Estimating manifold dimension for class {key} ...")
    id, _ = twoNN.estimate_dim(acts)
    # print(f"Estimated manifold dimension for class {key}: ", id)
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
  parser.add_argument('--dataset', type=str, help='Dataset to use (cifar, restricted_imagenet, imagenet)', default='imagenet')
  parser.add_argument('--arch', type=str, help='Model Architecture', default='resnet')
  parser.add_argument('--model_type', type=str, help='Type of model: standard, adv_trained or robust', default='standard')
  parser.add_argument('--task', type=str, help='Task to perform: acts or dims', default='acts')

  args = parser.parse_args()
  args.dataset = args.dataset.lower()
  
  assert args.dataset in ['imagenet'], "Invalid dataset" #! only supporting imagenet for now
  assert args.arch in ['resnet', 'vone_resnet'], "Model not supported"
  assert args.model_type in ['standard', 'adv_trained', 'robust'], "Invalid model type"
  assert args.task in ['acts', 'dims'], "Invalid task"
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if args.dataset == 'imagenet':    
    if args.task == 'acts':
      model_ext = ''
      model = None
      if args.arch == 'resnet':
        if args.model_type == 'standard':
          model_path = f'./models/{args.dataset}_r50{model_ext}_train.pt'
          model = get_model(arch='resnet', dataset=args.dataset, train_mode='standard', weights_path=model_path).to(device)
          model = model.to(device)

        elif args.model_type == 'adv_trained':
          model_ext = '_adv'
          model_path = f'./models/resnet50_l2_eps3.pt'
          model = get_model(arch='resnet', dataset=args.dataset, train_mode='adv_trained', weights_path=model_path).to(device)
          model = model.to(device)


      if args.arch == 'vone_resnet50':
       model_ext = '_vone'
       model = get_model(arch='vone_resnet', dataset=args.dataset, train_mode='standard', weights_path=None).to(device)
      
      model.eval()

      transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      
      val_dataset = datasets.ImageFolder(root='./data/imagenet/val', transform=transform)
      val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)
      class_activations = get_activations(model, val_loader, device, bs=32)
    
      print("Saving Class Activations ...") 
      
      save_path= f'./{args.dataset}_r50{model_ext}_train'
      if not os.path.exists(save_path):
        os.makedirs(save_path)

      with open(f'{save_path}/class_acts_test.pkl', 'wb') as f:
        pickle.dump(class_activations, f)
      return

    elif args.task == 'dims':
      model_ext = ''
      if args.model_type == 'adv_trained': model_ext = '_adv'
      elif args.model_type == 'vone_resnet': model_ext = '_vone'

      estimate_manifold_dim(model_ext, dataset_name=args.dataset)
      return


if __name__ == '__main__':
  main()





