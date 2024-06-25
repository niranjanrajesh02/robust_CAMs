import numpy as np
import torch
import tqdm

# tqdm
def tqdm_enumerate(iterator):
  for i, x in enumerate(tqdm.tqdm(iterator)):
    yield i, x

def get_activations(model, dl):
  activations = []
  print("Getting Activations ...")

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

def main(gpu=False):
  # import resnet50 from robustness lib
  import numpy as np
  import matplotlib.pyplot as plt
  import os

  # Load the model
  if gpu:
    from robustness import model_utils, datasets
    ds = datasets.CIFAR('')
    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path='cifar_nat.pt')
  else:
    # get from pytorch
    from torchvision.models import resnet50
    model = resnet50(weights=None)


  return


if __name__ == '__main__':
  main()





