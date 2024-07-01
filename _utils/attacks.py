import foolbox as fb
import numpy as np


def prepare_attack(model, attack_params):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    if attack_params['attack_type'] == 'L2_PGD': 
      eps = attack_params['epsilon']
      step_size = eps / 5
      iterations = attack_params['iterations']
      attack = fb.attacks.L2PGD(steps=iterations, abs_stepsize=step_size, random_start=True)

      return fmodel, attack

    

