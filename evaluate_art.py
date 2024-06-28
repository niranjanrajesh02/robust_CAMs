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


def get_classwise_acc(model, attack, test_loader, num_classes=1000):
  class_correct = {i: 0 for i in range(num_classes)}
  class_total = {i: 0 for i in range(num_classes)}

  print("Getting Classwise Accuracy ...")

  
  for inputs, labels in tqdm(test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels = inputs.to(device), labels.to(device)
    # inputs, labels = inputs.numpy(), labels.numpy()

    adv_images = attack.generate(x=inputs)
    # Generate adversarial examples
    adv_images_tensor = torch.tensor(adv_images)
    outputs = model._model(adv_images_tensor)
    _, preds = torch.max(outputs, 1)
    
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

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device: ", device)

  if args.model_type == 'adv_trained':
    model_ext = '_adv'
  elif args.model_type == 'standard':
    from torchvision.models import resnet50
    model_path = f'./models/{args.dataset}_r50{model_ext}_train.pt'
    model = resnet50(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    classifier = PyTorchClassifier(
      model=model,
      loss = torch.nn.CrossEntropyLoss(),
      optimizer = torch.optim.SGD(model.parameters(), lr=0.001),
      input_shape=(3, 224, 224),
      nb_classes=1000,
      clip_values=(0, 1)
    )
    print("Standard Resnet Loaded Successfully")
  elif args.model_type == 'vone_resnet':
    model_ext = '_vone'
    if args.dataset == 'imagenet':
      print("Loading VOneNet Model")
      model = vonenet.get_model(model_arch='resnet50', pretrained=True, noise_mode=None)
      print("VOneNet Loaded Successfully")
      model = model.to(device)
      classifier = PyTorchClassifier(
        model=model,
        loss = torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0, 1)
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
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
  
  classifier.to(device)
  attack = ProjectedGradientDescent(
    estimator=classifier,
    norm=2,
    eps=0.2,
    eps_step=0.01,
    max_iter=7,
    targeted=False
  )
  

  class_accuracies = get_classwise_acc(model, attack, val_loader)
  save_path= f'./{args.dataset}_r50{model_ext}_train'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  with open(f'./{save_path}/classwise_acc_e{args.eps}.pkl', 'wb') as f:
    pickle.dump(class_accuracies, f)

  return

if __name__ == '__main__':
  main()