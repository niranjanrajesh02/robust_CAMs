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
from robustness.datasets import CIFAR, RestrictedImageNet, ImageNet
import os
import argparse
import torch
import cox.store
import pickle
from tqdm import tqdm

def get_classwise_acc(m, test_loader, attack_kwargs, eps, ds_name='cifar'):
  if eps == 0:
    print("No attack being performed")
  else:
    print(f"Attack being performed with Epsilon: {eps}")
  
  num_classes = 10
  if ds_name == 'imagenet':
    num_classes = 1000
  elif ds_name == 'restricted_imagenet':
    num_classes = 9
  
  class_correct = {i: 0 for i in range(num_classes)}
  class_total = {i: 0 for i in range(num_classes)}

  # for param in m.model.parameters():
  #   param.requires_grad = False

  print("Getting Classwise Accuracy ...")

  for inputs, labels in tqdm(test_loader):
    inputs, labels = inputs.cuda(), labels.cuda()

    # Generate adversarial examples
    if eps != 0:
      torch.autograd.set_detect_anomaly(True)
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
  classwise_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)}

  return classwise_acc


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type', type=str, help='Type of model: standard, adv_trained or robust', default='standard')
  parser.add_argument('--eps', type=float, help='Epsilon value for adversarial training', default=0)
  parser.add_argument('--dataset', type=str, help='Dataset to use (cifar, restricted_imagenet, imagenet)', default='cifar')
  args = parser.parse_args()
  
  args.dataset = args.dataset.lower()
  
  attack_kwargs = {
      'constraint': '2',  # l2 constraint
      'eps': args.eps,  # epsilon value for l-inf
      'step_size': args.eps/5,  # step size for PGD
      'iterations': 7,  # number of iterations for PGD
      'do_tqdm': False
  }

  assert args.dataset in ['cifar', 'restricted_imagenet', 'imagenet'], "Invalid dataset"
  assert args.model_type in ['standard', 'adv_trained', 'robust', 'vone_resnet'], "Invalid model type"

  print("\n\n=============================================")
  print(f"Dataset: {args.dataset}, Model Type: {args.model_type}, Epsilon: {args.eps}")
  print("=============================================")
  model_ext = ''
  if args.model_type == 'adv_trained':
    model_ext = '_adv'
  elif args.model_type == 'robust':
    model_ext = '_robust'
  elif args.model_type == 'vone_resnet':
    model_ext = '_vone_resnet'


  if args.dataset == 'cifar':
    ds = CIFAR('./data')
  elif args.dataset == 'restricted_imagenet':
    ds = RestrictedImageNet('./data/imagenet')
  elif args.dataset == 'imagenet':
    ds = ImageNet('./data/imagenet')
  print("Dataset Found. Loading Model ...")

  if args.model_type == 'vone_resnet' and args.dataset == 'imagenet':
    import vonenet
    print("Loading VOneNet Model")
    v1_model = vonenet.get_model(model_arch='resnet50', pretrained=True)
    print("VOneNet Model Loaded Successfully from Vonenet, now loading into Robustness Library")
    model, _ = model_utils.make_and_restore_model(arch=v1_model.module, dataset=ds, add_custom_forward=True)
    print("Model Loaded Successfully")
  elif args.model_type != 'vone_resnet':
    if args.dataset  == 'cifar':
      model_path = f'./cifar_r50{model_ext}_train/checkpoint.pt.latest'
    else:
      model_path = f'./models/{args.dataset}_r50{model_ext}_train.pt'
    
    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=model_path)
    print("Model Loaded Successfully")
    
  test_loader = ds.make_loaders(batch_size=256, workers=1, only_val=True)[1]
  print("Test Loader Created")

  model.eval()
  classwise_acc = get_classwise_acc(model, test_loader, attack_kwargs, args.eps, ds_name=args.dataset)
  print("Classwise Accuracy: ", classwise_acc)

  # store classwise accuracy as a pickle file
  save_path= f'./{args.dataset}_r50{model_ext}_train'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  with open(f'{save_path}/classwise_acc_e{args.eps}.pkl', 'wb') as f:
    pickle.dump(classwise_acc, f)
  
  print(f"Classwise Accuracy stored as pickle file for model type: {args.model_type} and dataset: {args.dataset} with epsilon: {args.eps}")

  return

if __name__ == '__main__':
  main()