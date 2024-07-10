import numpy as np
import torch
from tqdm import tqdm
import os
import sys
import torch.hub
import os
import argparse
import pickle
from _utils.id_est import estimate_twonn_dim, estimate_pca_dim
from _utils.model import get_model
from _utils.data import get_dataloader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda.amp import autocast

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
  
  with autocast():
    _ = model(input)

  h1.remove()
  # print("Activations shape:", acts.shape)
  activations_arr = np.array(activations[0])
  return activations_arr

def get_activations(model, dl, device, bs=1):    
  num_classes = 1000

  class_activations = {i: [] for i in range(num_classes)}

  print("Getting Class Activations ...")

  # batches = 0
  for inputs, labels in tqdm(dl):
    inputs, labels = inputs.to(device), labels.to(device)
    # get class index and corresponding activations !
    activations = get_layer_accs(model, inputs)
    # print("Activations Shape: ", activations.shape)
    # print("Labels Shape: ", labels.shape)
    # append to class activations

    # iterate through batch
    for i in range(len(labels)):
      label = labels[i].item()
      class_activations[label].append(activations[i])
    # batches += 1
    # if batches == 100:
    #   break
    

  print("Class Activations obtained.")

  for key in class_activations:
    class_activations[key] = np.array(class_activations[key], dtype=np.float16)
    # print(f"Class {key} Activations Shape: ", class_activations[key].shape)
  print("Finished getting class activations")
  return class_activations

def estimate_manifold_dim(model_ext, dataset_name='imagenet', data_split='val'):
  print("Estimating Manifold Dimension ...")
  class_acts_file = f'./{dataset_name}_r50{model_ext}_train/class_acts_{data_split}.pkl'
  print("Class Acts File: ", class_acts_file)
  # load class activations
  if os.path.exists(class_acts_file):
    with open(class_acts_file, 'rb') as f:
      class_activations = pickle.load(f)
  else:
    print("Class Activations File not found.")
    return
  
  class_dims_2nn = {i : 0 for i in range(1000)}
  class_dims_pca = {i : 0 for i in range(1000)}

  for key in tqdm(class_activations):
    acts = np.array(class_activations[key])
    # print(f"Estimating manifold dimension for class {key} ...")
    id_2nn, _ = estimate_twonn_dim(acts)
    id_pca = estimate_pca_dim(acts)

    class_dims_2nn[key] = id_2nn
    class_dims_pca[key] = id_pca
  

 
  # class_dims['all'] = 0
  # # concatenate all class activations
  # all_acts = [class_activations[key] for key in class_activations]
  # all_acts = np.concatenate(all_acts, axis=0)
  # print("Estimating manifold dimension for all classes with concatenated activations: ", all_acts.shape)
  # id, _ = id_est.estimate_dim(all_acts)

  # print("Estimated manifold dimension for all classes: ", id)
  # class_dims['all'] = id
  

  return class_dims_2nn, class_dims_pca

def main():
  # TODO : Add support for other archs
  parser = argparse.ArgumentParser(description='Get Class Activations')
  parser.add_argument('--dataset', type=str, help='Dataset to use (cifar, restricted_imagenet, imagenet)', default='imagenet')
  parser.add_argument('--arch', type=str, help='Model Architecture', default='resnet')
  parser.add_argument('--model_type', type=str, help='Type of model: standard, adv_trained or robust', default='standard')
  parser.add_argument('--task', type=str, help='Task to perform: acts or dims', default='acts')
  parser.add_argument('--data_split', type=str, help='Data split to use: train or val', default='val')

  args = parser.parse_args()
  args.dataset = args.dataset.lower()
  
  assert args.dataset in ['imagenet'], "Invalid dataset" 
  assert args.arch in ['resnet', 'vone_resnet'], "Model not supported"
  assert args.model_type in ['standard', 'adv_trained', 'robust'], "Invalid model type"
  assert args.task in ['acts', 'dims'], "Invalid task"
  assert args.data_split in ['train', 'val'], "Invalid data split"
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if args.dataset == 'imagenet':    
    if args.task == 'acts':
      model_ext = ''
      model = None
      if args.arch == 'resnet':
        if args.model_type == 'standard':
          model_path = f'./models/{args.dataset}_r50{model_ext}_train.pt'
          model = get_model(arch='resnet50', dataset=args.dataset, train_mode='standard').to(device)
          model = model.to(device)

        elif args.model_type == 'adv_trained':
          model_ext = '_adv'
          model_path = f'./models/resnet50_l2_eps3.pt'
          model = get_model(arch='resnet50', dataset=args.dataset, train_mode='adv_trained').to(device)
          model = model.to(device)


      if args.arch == 'vone_resnet50':
       model_ext = '_vone'
       model = get_model(arch='vone_resnet', dataset=args.dataset, train_mode='standard', weights_path=None).to(device)
      
      model.eval()

      batch_size = 32
      dl = get_dataloader(ds_name=args.dataset, split=args.data_split, bs=batch_size)
      class_activations = get_activations(model, dl, device, bs=batch_size)
    
      print("Saving Class Activations ...") 
      
      save_path= f'./{args.dataset}_r50{model_ext}_train'
      if not os.path.exists(save_path):
        os.makedirs(save_path)

      with open(f'{save_path}/class_acts_{args.data_split}.pkl', 'wb') as f:
        pickle.dump(class_activations, f)
      return

    elif args.task == 'dims':
      model_ext = ''
      if args.model_type == 'adv_trained': model_ext = '_adv'
      elif args.model_type == 'vone_resnet': model_ext = '_vone'

      class_dims_2nn, class_dims_pca =  estimate_manifold_dim(model_ext, dataset_name=args.dataset, data_split=args.data_split)

      with open(f'./{args.dataset}_r50{model_ext}_train/class_dims_2nn_{args.data_split}.pkl', 'wb') as f:
        pickle.dump(class_dims_2nn, f)

      with open(f'./{args.dataset}_r50{model_ext}_train/class_dims_pca_{args.data_split}.pkl', 'wb') as f:
        pickle.dump(class_dims_pca, f)

      print("Manifold Dimensions Saved Successfully.")

      return


if __name__ == '__main__':
  main()





