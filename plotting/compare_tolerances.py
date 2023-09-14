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

    tolerance_options = ["t0_nd1", "t0_nd5", "t0_nd10",
                         "t0-05_nd1", "t0-05_nd5", "t0-05_nd10",
                         "t0-1_nd1", "t0-1_nd5", "t0-1_nd10"]

    for t in tolerance_options:
        folders = [f"results/tolerances_2/jsrlgs/*{t}_antmaze*.txt", f"results/tolerances_2/jsrl/*{t}_antmaze*.txt"]

        eval_returns = [glob.glob(folder) for folder in folders]
        algos = ["JSRL-GS", "JSRL"]

        assert len(algos) == len(folders), f"Num folders {len(folders)} != num algorithm names {len(algos)}"
        print("Loading data...")
        try:
            all_data = polars_read(algos, eval_returns)
        except ValueError:
            print(f"Folders in {folders} are empty or do not exist.")
            exit()
        print("Data loaded. Making plots..")

        datas = []
        for d in all_data.partition_by("N data", "Algo"):
            d = d.with_columns(d["Return"].ewm_mean(alpha=0.5))
            datas.append(d)
        all_data = pl.concat(datas)

        ax = sns.lineplot(x="Time Step", y="Return", hue="Algo",
                            style="N data", data=all_data)
        ax.axvline(0, color="grey", linestyle="dashed", alpha=0.8)
        ax.text(-0.5e6, all_data["Return"].mean()+2*all_data["Return"].std(), "Offline Training", ha='center', size=10)
        ax.text(0.5e6, all_data["Return"].mean()+2*all_data["Return"].std(), "Online Fine Tuning", ha='center', size=10)
        print("Plots made. Saving plots...")
        plt.tight_layout()
        plt.savefig(f"results/tolerances_2/{t}.png")
        print(t)
        #plt.show()
        plt.close()
        #break

