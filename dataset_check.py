import pickle
import glob
import pandas as pd
import h5py
import numpy as np

for file in glob.glob("datasets/flappy_ppo_1000000.pkl"):
    with open(file, "rb") as f:
        data = pd.read_pickle(file)

def convert_file(fn):
    f = h5py.File(fn, 'r')
    df_dict = {}
    df = pd.DataFrame()
    for k in f.keys():
        if k == "infos": continue
        df_dict[k] = np.array(f[k])
        print(f"{k}: {df_dict[k].shape}")
        if k == "observations":
            next_obs = np.zeros(df_dict[k].shape)
            next_obs[:-1] = df_dict[k][1:]
            df_dict["next_observations"] = next_obs
    with open(fn[:-5]+".pkl", 'wb') as handle:
        pickle.dump(df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    #for fn in glob.glob("datasets/*hdf5"):
    fn = "datasets/antmaze-umaze-v2.hdf5"
    convert_file(fn)
    for file in glob.glob(fn[:-5]+".pkl"):
        with open(file, "rb") as f:
            data = pd.read_pickle(file)
            print(data)