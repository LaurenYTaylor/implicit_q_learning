import pickle
import glob
import pandas as pd

for file in glob.glob("datasets/*.pkl"):
    with open(file, "rb") as f:
        data = pd.read_pickle(file)
        print(data)
