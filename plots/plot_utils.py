import pickle

data_path = '../data/'

def load_data(model_ext, data_ext):
    with open(f"{data_path}_{model_ext}/{data_ext}.pkl", "rb") as f:
        return pickle.load(f)