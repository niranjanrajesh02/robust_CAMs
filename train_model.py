import sys
import torch.hub
import os
from torchvision import transforms
# Define the function
def load_state_dict_from_url(*args, **kwargs):
    return torch.hub.load_state_dict_from_url(*args, **kwargs)

# Create a dummy module
class DummyModule:
    def __init__(self):
        self.load_state_dict_from_url = load_state_dict_from_url

# Replace the faulty import
sys.modules['torchvision.models.utils'] = DummyModule()


from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR, DataSet
from robustness.data_augmentation import TRAIN_TRANSFORMS_DEFAULT, TEST_TRANSFORMS_DEFAULT
from torch.utils.data import Dataset, DataLoader, TensorDataset

from cox.utils import Parameters
import cox.store
from cox import utils
from cox import store

NUM_WORKERS = 1
BATCH_SIZE = 128

def init_model_data():
  ds = CIFAR('./data')
  m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
  train_loader, val_loader = ds.make_loaders(batch_size=BATCH_SIZE, workers=NUM_WORKERS, data_aug=False)

  return m, train_loader, val_loader

def train_model(adv_train=False, m=None, train_loader=None, val_loader=None):
  out_path = "cifar_r50_train"
  if adv_train:
      out_path = "cifar_r50_adv_train"
  

  # Hard-coded base parameters - https://robustness.readthedocs.io/en/latest/api/robustness.defaults.html#module-robustness.defaults
  train_kwargs = {
      'out_dir': out_path,
      'adv_train': 0,
      'adv_eval': 0,
      'epochs': 150,
      'batch_size': 128,
      'weight_decay': 5e-4,
      'step_lr': 50,
      'lr': 0.1,
      'momentum': 0.9,
      'constraint': '2',
      'eps': 0.5,
      'attack_lr': 0.1, #step size of attack
      'attack_steps': 7,
      'save_ckpt_iters': -1
  }

  if adv_train:
      train_kwargs['adv_train'] = 1
  
  train_args = Parameters(train_kwargs)

  # Fill whatever parameters are missing from the defaults
  train_args = defaults.check_and_fill_args(train_args,
                          defaults.TRAINING_ARGS, CIFAR)
  train_args = defaults.check_and_fill_args(train_args,  defaults.PGD_ARGS, CIFAR)

  store = store.Store(out_path, exp_id=f'{out_path}_store')
  train.train_model(train_args, m, (train_loader, val_loader), store=store)
  return


class CustomTensorDataset(DataSet):
    def __init__(self, train_set,test_set, num_classes=10):
        # Define data mean and std (example values; adjust as necessary)
        data_mean = torch.tensor([0.4914, 0.4822, 0.4465])
        data_std = torch.tensor([0.2023, 0.1994, 0.2010]) 

        
        # Define transforms
        transform_train = TRAIN_TRANSFORMS_DEFAULT(32)
        transform_test = TEST_TRANSFORMS_DEFAULT(32)

        # Call the superclass __init__ method with appropriate arguments
        super(CustomTensorDataset, self).__init__(
            name='custom',
            num_classes=num_classes,
            mean=data_mean,
            std=data_std,
            custom_class=None,
            label_mapping=None,
            transform_train=transform_train,
            transform_test=transform_test
        )

        # Set the dataset tensors after splitting with sklearn train_test_split
        self.train_dataset = train_set
        self.test_dataset = test_set
        
    def get_model(self, arch, pretrained):
        from robustness import cifar_models # or cifar_models
        assert not pretrained, "pretrained only available for ImageNet"
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

    def make_loaders(self, batch_size, workers, only_val=False):
        train_loader = None
        if not only_val:
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        return train_loader, test_loader

def robust_train():
    out_path = "cifar_r50_robust_train"

    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
   
    data_path = './data/d_robust_cifar'

    ims = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_ims")))
    labs = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_lab")))
    dataset = TensorDataset(ims, labs) # !! does not use transforms

    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    #  custom dataset with robustness library
    ds = CustomTensorDataset(train_set, test_set)
    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=BATCH_SIZE, workers=NUM_WORKERS, only_val=True)
    
    train_kwargs = {
      'out_dir': out_path,
      'adv_train': 0,
      'adv_eval': 0,
      'epochs': 150,
      'batch_size': 128,
      'weight_decay': 5e-4,
      'step_lr': 50,
      'lr': 0.1,
      'momentum': 0.9,
      'constraint': '2',
      'eps': 0.5,
      'attack_lr': 0.1, #step size of attack
      'attack_steps': 7,
      'save_ckpt_iters': -1
    }

    train_args = Parameters(train_kwargs)

    # Fill whatever parameters are missing from the defaults
    train_args = defaults.check_and_fill_args(train_args,
                            defaults.TRAINING_ARGS, CIFAR)
    train_args = defaults.check_and_fill_args(train_args,  defaults.PGD_ARGS, CIFAR)

    store = store.Store(out_path, exp_id=f'{out_path}_store')
    train.train_model(train_args, model, (train_loader, val_loader), store=store)

    return


def main():
  import argparse
  parser = argparse.ArgumentParser(description='Train a model')
  parser.add_argument('--train_mode',  help='Training Mode: standard, adv or robust', type=str, default='standard')
  args = parser.parse_args()

  assert args.train_mode in ['standard', 'adv', 'robust'], "Invalid training mode"
  adv_train=0
  if args.train_mode == 'adv':
    adv_train = 1
  
  

  print("Training")
  
  if args.train_mode == 'robust':
     robust_train()
  else:
    m, train_loader, val_loader = init_model_data()
    train_model(adv_train=adv_train, m=m, train_loader=train_loader, val_loader=val_loader)
  print("Done training")

if __name__ == '__main__':
  main()