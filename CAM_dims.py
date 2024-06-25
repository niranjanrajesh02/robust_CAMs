import numpy as np
import torch

def get_activations(model, dl):
  activations = []

  def activation_hook(module, input, output):
    activations.append(output.detach().cpu().numpy().reshape(output.shape[0], -1))
    return

  model.model.layer4.register_forward_hook(activation_hook)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model.eval()
  with torch.no_grad():
    for images, labels, in dl:
      images = images.to(device)
      labels = labels.to(device)
      _ = model(images)

  return np.concatenate(activations, axis=0)

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





