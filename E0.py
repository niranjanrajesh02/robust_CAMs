import sys
import torch.hub

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
from robustness.datasets import CIFAR
from cox.utils import Parameters
import cox.store
from cox import utils
from cox import store

NUM_WORKERS = 10
BATCH_SIZE = 128
ds = CIFAR('./data')
m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
train_loader, val_loader = ds.make_loaders(batch_size=BATCH_SIZE, workers=NUM_WORKERS, data_aug=False)
out_store = cox.store.Store('./')

# Hard-coded base parameters - https://robustness.readthedocs.io/en/latest/api/robustness.defaults.html#module-robustness.defaults
train_kwargs = {
    'out_dir': "train_out",
    'adv_train': 0,
    'adv_eval': 0,
    'epochs': 2,
    'batch_size': 128,
    'weight_decay': 5e-4,
    'step_lr': 50,
    'lr': 0.1,
    'momentum': 0.9,
    'constraint': 2,
    'eps': 0.5,
    'attack_lr': 0.1,
    'attack_steps': 7,
    'save_ckpt_iters': 10
}
train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, CIFAR)
train_args = defaults.check_and_fill_args(train_args,  defaults.PGD_ARGS, CIFAR)

def main():
  print("Training")
  train.train_model(train_args, m, (train_loader, val_loader), store=out_store)
  print("Done training")

if __name__ == '__main__':
  main()