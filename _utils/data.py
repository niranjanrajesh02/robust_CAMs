from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
import shutil
import shlex
import urllib

def get_dataloader(ds_name='imagenet', split='val', bs=32):
  assert ds_name in ['imagenet'], "Dataset not supported"
  assert split in ['train', 'val'], "Split has to be either train or val"
  
  data_path = f'/scratch/venkat/niranjan/{ds_name}/{split}'
  print(f'Getting dataloader for {ds_name}/{split} from {data_path}')
  if not os.path.exists(data_path):
    raise ValueError("Data path does not exist. Please download the dataset")

  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  dataset = ImageFolder(root=data_path, transform=transform)
  loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)

  return loader




