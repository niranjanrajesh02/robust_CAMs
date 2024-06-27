import sys
import torch
def load_state_dict_from_url(*args, **kwargs):
    return torch.hub.load_state_dict_from_url(*args, **kwargs)

# Create a dummy module
class DummyModule:
    def __init__(self):
        self.load_state_dict_from_url = load_state_dict_from_url

# Replace the faulty import
sys.modules['torchvision.models.utils'] = DummyModule()

from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import os
ds = CIFAR('./data')
import argparse
import torch
import cox.store
import pickle
from tqdm import tqdm

def get_classwise_acc(m, test_loader, attack_kwargs, eps):
  if eps == 0:
    print("No attack being performed")
  else:
    print(f"Attack being performed with Epsilon: {eps}")
  
  class_correct = {i: 0 for i in range(10)}
  class_total = {i: 0 for i in range(10)}

  for param in m.model.parameters():
    param.requires_grad = False

  print("Getting Classwise Accuracy ...")

  for inputs, labels in tqdm(test_loader):
    inputs, labels = inputs.cuda(), labels.cuda()

    # Generate adversarial examples
    if eps != 0:
      _, adv_in = m(inputs, labels, make_adv=True, **attack_kwargs)
      out, _ = m(adv_in)
      preds = torch.argmax(out, dim=1)

    else:
      out, _ = m(inputs, labels, make_adv=False)
      preds = torch.argmax(out, dim=1)
  
    # Update classwise accuracy for the batch
    for i in range(len(labels)):
      label = labels[i].item()
      pred = preds[i].item()
      class_correct[label] += int(pred == label)
      class_total[label] += 1

  # Calculate classwise accuracy
  classwise_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)}

  return classwise_acc


def main():
  parser = argparse.ArgumentParser(description='Train a model on CIFAR')
  parser.add_argument('--model_type', type=str, help='Type of model: standard, adv_trained or robust', default='standard')
  parser.add_argument('--eps', type=float, help='Epsilon value for adversarial training', default=0.5)
  args = parser.parse_args()

  
  attack_kwargs = {
      'constraint': '2',  # l-inf constraint
      'eps': args.eps,  # epsilon value for l-inf
      'step_size': 0.1,  # step size for PGD
      'iterations': 10,  # number of iterations for PGD
      'do_tqdm': False
  }

  assert args.model_type in ['standard', 'adv_trained', 'robust'], "Invalid model type"
  print("\n\n=============================================")
  print(f"Model Type: {args.model_type}, Epsilon: {args.eps}")
  model_ext = ''
  if args.model_type == 'adv_trained':
    model_ext = '_adv'
  elif args.model_type == 'robust':
    model_ext = '_robust'

  model_path = f'/home/venkat/niranjan/robust_CAMs/cifar_r50{model_ext}_train/checkpoint.pt.latest'
  print(f"Trying to load Model at Path: {model_path}")

  model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=model_path)
  print("Model Loaded Successfully")
  
  test_loader = ds.make_loaders(batch_size=10, workers=1, only_val=True)[1]
  model.eval()


  classwise_acc = get_classwise_acc(model, test_loader, attack_kwargs, args.eps)
  print("Classwise Accuracy: ", classwise_acc)

  # store classwise accuracy as a pickle file
  with open(f'/home/venkat/niranjan/robust_CAMs/cifar_r50{model_ext}_train/classwise_acc_e{args.eps}.pkl', 'wb') as f:
    pickle.dump(classwise_acc, f)
  
  print("Classwise Accuracy stored as pickle file")

  
  return

if __name__ == '__main__':
  main()