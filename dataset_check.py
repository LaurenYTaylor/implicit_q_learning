import pickle
import glob
import pandas as pd

for file in glob.glob("datasets/flappy_ppo_1000000.pkl"):
    with open(file, "rb") as f:
        data = pd.read_pickle(file)
