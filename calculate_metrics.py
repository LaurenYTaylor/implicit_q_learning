import glob
import numpy as np
import polars as pl
import re

file_stub = "results/all_var_at_31023"
algo = "jsrlgs"
env = "antmaze"

filenames = glob.glob(f"{file_stub}/{algo}/*{env}*.txt")

def polars_read(algo, filenames):
    datas = []
    for k, fname in enumerate(filenames):
        data = pl.read_csv(fname, separator=" ", has_header=False, new_columns=["Time Step", "Return"])
        seed = int(re.search("_s([0-9]+)", fname).group(1))
        data_size = int(re.search("_d([0-9]+)", fname).group(1))

        data = data.with_columns([pl.Series([algo]*len(data)).alias("Algo"),
                                 pl.Series([seed]*len(data)).alias("Seed"),
                                 pl.Series([data_size]*len(data)).alias("N data")])
        datas.append(data)
    return pl.concat(datas)

data = polars_read(algo, filenames)

for dsize in data.partition_by("N data"):
    pretraining = dsize.filter((pl.col("Time Step")==0))["Return"]
    pretraining_mean = pretraining.mean()
    pretraining_std = pretraining.std()

    min_rets = []
    final_rets = []
    for seed in dsize.partition_by("Seed"):
        #print(seed.filter((pl.col("Time Step")>0)))
        min_rets.append(seed.filter((pl.col("Time Step")>0)).select(pl.min("Return")))
    #print(min_rets)
    online_mean = np.mean(min_rets)
    online_std = np.std(min_rets)

    final = dsize.filter((pl.col("Time Step")==1000000))["Return"]
    final_mean = final.mean()
    final_std = final.std()

    print(f"N data: {dsize['N data'][0]},"\
            f" eta={np.round(pretraining_mean-online_mean,2)} +/- {np.round(.5*np.sqrt(online_mean**2/5+pretraining_mean**2/5),2)}"\
            f" Final: {np.round(final_mean,2)} +\- {np.round(final_std,2)}")