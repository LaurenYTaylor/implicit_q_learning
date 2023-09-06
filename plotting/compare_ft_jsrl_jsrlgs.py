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

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    folders = ["results/flappy/0/*s[01234]*flappy_heuristic*.txt", "results/flappy/1/*.txt","results/flappy/2/*.txt"]

    eval_returns = [glob.glob(folder) for folder in folders]
    algos = ["JSRL-GS", "JSRL", "FT"]

    assert len(algos) == len(folders), f"Num folders {len(folders)} != num algorithm names {len(algos)}"

    print("Loading data...")
    try:
        all_data = polars_read(algos, eval_returns)
    except ValueError:
        print(f"Folders in {folders} are empty or do not exist.")
        exit()
    print("Data loaded. Making plots..")

    ax = sns.lineplot(x="Time Step", y="Return", hue="Algo",
                        style="N data", data=all_data)
    ax.axvline(0, color="grey", linestyle="dashed", alpha=0.8)
    ax.text(-0.5e6, all_data["Return"].mean()+2*all_data["Return"].std(), "Offline Training", ha='center', size=10)
    ax.text(0.5e6, all_data["Return"].mean()+2*all_data["Return"].std(), "Online Fine Tuning", ha='center', size=10)
    print("Plots made. Saving plots...")
    #plt.savefig("plotting/plots/wtf.png")
    plt.show()
