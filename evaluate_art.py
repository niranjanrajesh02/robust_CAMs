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
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import pickle
import os
import numpy as np
from torchvision.models import resnet50

def get_classwise_acc(model, attack, eps, test_loader, num_classes=1000):
  class_correct = {i: 0 for i in range(num_classes)}
  class_total = {i: 0 for i in range(num_classes)}

  print("Getting Classwise Accuracy ...")

  
  for inputs, labels in tqdm(test_loader):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    inputs, labels = inputs.to(device), labels.to(device)

    if eps != 0:
      adv_images = attack.generate(x=inputs)
      # Generate adversarial examples
      adv_images_tensor = torch.tensor(adv_images).to(device)
      outputs = model.predict(adv_images_tensor)
      preds = np.argmax(outputs, axis=1)

    else:
      outputs = model.predict(inputs)
      preds = np.argmax(outputs, axis=1)
    
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
  # * Loading the Model
  model_ext = ''
  model = None

  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  print("Device: ", device)

  model = None
  if args.model_type == 'adv_trained':
    model_ext = '_adv'
    model_path = f'./models/{args.dataset}_r50{model_ext}_train.pt'

    from robustness import model_utils
    from robustness.datasets import ImageNet
    ds = ImageNet('data/imagenet')
    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=model_path)
    model = model.model.to(device)

    print("Adversarially Trained Resnet Loaded Successfully")

  elif args.model_type == 'standard':
    
    model_path = f'./models/{args.dataset}_r50{model_ext}_train.pt'
    model = resnet50(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    print("Standard Resnet Loaded Successfully")

  elif args.model_type == 'vone_resnet':
    model_ext = '_vone'
    if args.dataset == 'imagenet':
      print("Loading VOneNet Model")
      model = vonenet.get_model(model_arch='resnet50', pretrained=True, noise_mode=None).to(device)
      print("VOneNet Loaded Successfully")
      # model = model.to(device)
    else:
      print("VOneNet not available for this dataset")
      return
  

  assert model is not None, "Model not loaded successfully"
  model.eval()
  
  device_str = 'cpu'
  classifier = PyTorchClassifier(
        model=model,
        loss = torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0, 1),
        device_type=device_str
      )

  print("Model compiled successfully as ART Classifier")

  val_loader = None
  #* Loading the dataset
  if args.dataset == 'imagenet':
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(root='./data/imagenet/val', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)
  
  attack = ProjectedGradientDescent(
    estimator=classifier,
    norm=2,
    eps=args.eps,
    eps_step=0.01,
    max_iter=7,
    targeted=False
  )

  class_accuracies = get_classwise_acc(classifier, attack, args.eps, val_loader)
  save_path= f'./{args.dataset}_r50{model_ext}_train'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  print("Classwise Accuracies: ", class_accuracies)
  print(f"Saving Classwise Accuracies to {save_path}")

  with open(f'./{save_path}/classwise_acc_e{args.eps}.pkl', 'wb') as f:
    pickle.dump(class_accuracies, f)

  return

if __name__ == '__main__':
  main()