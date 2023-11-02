import pickle
import glob
import pandas as pd
import numpy as np

for file in glob.glob("datasets/flappy_ppo*.pkl"):
    with open(file, "rb") as f:
        print(file)
        data = pd.read_pickle(file)
        i = np.where(data['terminals'])[0]
        sub = np.concatenate(([0], i))
        i = np.concatenate((i, [i[-1]]))
        ep_lens = i-sub
        print(f"Ep len: {np.round(np.mean(ep_lens),2)}+\-{np.round(np.std(ep_lens),2)}")
        i=50
        for k, v in data.items():
            print(f"{k}: {v[:i]}")
