import foolbox as fb
import numpy as np


def prepare_attack(model, attack_params, transforms=None):
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    if attack_params['attack_type'] == 'L2_PGD': 
      eps = attack_params['epsilon']
      step_size = eps / 5
      iterations = attack_params['iterations']
      attack = fb.attacks.L2PGD(steps=iterations, abs_stepsize=step_size, random_start=True)

      return fmodel, attack

    

