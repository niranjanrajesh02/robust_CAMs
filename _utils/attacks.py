import foolbox as fb
import numpy as np
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim

def prepare_attack(model, attack_params):
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=None)
    if attack_params['attack_type'] == 'L2_PGD': 
      eps = attack_params['epsilon']
      step_size = eps / 5
      iterations = attack_params['iterations']
      attack = fb.attacks.L2PGD(steps=iterations, abs_stepsize=step_size, random_start=True)
  
      return fmodel, attack

    
def prepare_art_attack(model, attack_params, arch='vone_resnet'):
  assert arch in ['vone_resnet'], "Model not supported for ART attacks"

  mean = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
  std = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))

  classifier = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            preprocessing=(mean, std),
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            input_shape=(3, 224, 224),
            nb_classes=1000,
        )

  attack = ProjectedGradientDescent(
            estimator=classifier, 
            norm= 2,
            max_iter=attack_params['iterations'], 
            eps=attack_params['epsilon'], 
            eps_step= attack_params['epsilon'] / 5,
            targeted=False,
        )

  return classifier, attack