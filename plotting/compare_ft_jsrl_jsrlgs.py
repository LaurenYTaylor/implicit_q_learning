import matplotlib.pyplot as plt
import numpy as np
import tensorboard as tb
import glob
import seaborn as sns
import re
import polars as pl

def polars_read(algos, eval_returns):
    datas = []
    for i, algo in enumerate(algos):
        returns = eval_returns[i]
        for ret in returns:
            data = pl.read_csv(ret, separator=" ", has_header=False, new_columns=["Time Step", "Return"])

            seed = int(re.search("_s([0-9]+)", ret).group(1))
            data_size = int(re.search("_d([0-9]+)", ret).group(1))
            data = data.with_columns([pl.Series([algo]*len(data)).alias("Algo"),
                                     pl.Series([seed]*len(data)).alias("Seed"),
                                     pl.Series([data_size]*len(data)).alias("N data")])
            datas.append(data)
    return pl.concat(datas)

sns.set_theme(style="darkgrid")
folders = ["../logs/antmaze-umaze-v0_9/*.txt", "../logs/antmaze-umaze-v0_8_jsrl_ft/*.txt",
           "../logs/antmaze-umaze-v0_10_jsrlgs_ft/*.txt"]

eval_returns = [glob.glob(folder) for folder in folders]
algos = ["IQL", "JSRL", "JSRL-GS"]

print("Reading data...")
all_data = polars_read(algos, eval_returns)
print("Data loaded. Making plots..")

ax = sns.lineplot(x="Time Step", y="Return", hue="Algo",
                    style="N data", data=all_data)
ax.axvline(0, color="grey", linestyle="dashed", alpha=0.8)
ax.text(-0.5e6, all_data["Return"].mean()+2*all_data["Return"].std(), "Offline Training", ha='center', size=10)
ax.text(0.5e6, all_data["Return"].mean()+2*all_data["Return"].std(), "Online Fine Tuning", ha='center', size=10)
print("Plots made. Saving plots...")
plt.savefig("plots/partial_coldstart.png")
plt.show()
