from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import os
ds = CIFAR('./data')
import argparse
import torch
import cox.store
import pickle

EPS = 0.25

attack_kwargs = {
    'constraint': '2',  # l-inf constraint
    'eps': EPS,  # epsilon value for l-inf
    'step_size': 0.1,  # step size for PGD
    'iterations': 10,  # number of iterations for PGD
    'do_tqdm': True
}


def get_classwise_acc(m, test_loader):
  class_correct = {i: 0 for i in range(10)}
  class_total = {i: 0 for i in range(10)}

  for param in m.model.parameters():
    param.requires_grad = False

  for inputs, labels in test_loader:
    inputs, labels = inputs.cuda(), labels.cuda()

    # Generate adversarial examples
    _, adv_in = m(inputs, labels, make_adv=True, **attack_kwargs)
    preds, _ = m(adv_in)

    adv_preds = torch.argmax(preds, dim=1)

    # Update classwise accuracy for the batch
    for i in range(len(labels)):
      label = labels[i].item()
      pred = adv_preds[i].item()
      class_correct[label] += int(pred == label)
      class_total[label] += 1

  # Calculate classwise accuracy
  classwise_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)}

  return classwise_acc


def main():
  parser = argparse.ArgumentParser(description='Train a model on CIFAR')
  parser.add_argument('--model_type', type=str, help='Type of model: standard, adv_trained or robust', default='standard')

  args = parser.parse_args()
  assert args.model_type in ['standard', 'adv_trained', 'robust'], "Invalid model type"
  
  model_ext = ''
  if args.model_type == 'adv_trained':
    model_ext = 'adv_train'
  elif args.model_type == 'robust':
    model_ext = 'robust'

  model_path = f'/home/venkat/niranjan/robust_CAMs/cifar_r50_{model_ext}/checkpoint.pt.latest'

  if not os.path.exists(model_path):
    print("Model path does not exist")
    return
  elif not model_path.endswith('.pt'):
    print("Model path must be a .pt file")
    return
  

  model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=model_path)
  
  test_loader = ds.make_loaders(batch_size=10, workers=4, only_val=True)[1]
  model.eval()

  classwise_acc = get_classwise_acc(model, test_loader)
  print("Classwise Accuracy: ", classwise_acc)

  # store classwise accuracy as a pickle file
  with open(f'/home/venkat/niranjan/robust_CAMs/cifar_r50_{model_ext}/classwise_acc_e{EPS}.pkl', 'wb') as f:
    pickle.dump(classwise_acc, f)
  
  print("Classwise Accuracy stored as pickle file")

  
  return

if __name__ == '__main__':
  main()