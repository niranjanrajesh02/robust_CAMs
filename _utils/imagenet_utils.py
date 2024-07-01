import pandas as pd
import sys
from imagenet_stubs.imagenet_2012_labels import label_to_name

current_path = sys.path[0]
def get_class_label(ind):

  return(label_to_name(ind))

if __name__ == '__main__':
  get_class_label(0)