import matplotlib.pyplot as plt
import numpy as np
import tensorboard as tb
import glob
import seaborn as sns
import re
import polars as pl

def polars_read(algos, folder_names, at=False):
    datas = []
    for i, algo in enumerate(algos):
        folder = folder_names[i]
        if at:
            col_str = "Agent Type"
        else:
            col_str = "Return"
        for k, fname in enumerate(folder):
            data = pl.read_csv(fname, separator=" ", has_header=False, new_columns=["Time Step", col_str])
            if at:
                data = pl.concat([pl.DataFrame({"Time Step": -int(1e6), "Agent Type": float(0)}), data])
                data = pl.concat([pl.DataFrame({"Time Step": 0, "Agent Type": float(0)}), data])
            seed = int(re.search("_s([0-9]+)", fname).group(1))
            data_size = int(re.search("_d([0-9]+)", fname).group(1))

            data = data.with_columns([pl.Series([algo]*len(data)).alias("Algo"),
                                     pl.Series([seed]*len(data)).alias("Seed"),
                                     pl.Series([data_size]*len(data)).alias("N data")])
            datas.append(data)
    return pl.concat(datas)

def add_ax_decorations(ax):
    ax.axvline(0, color="grey", linestyle="dashed", alpha=0.8)
    ax.text(0.25, 1, "Offline\nTraining",
                        ha='center', va='bottom', size=10, transform=ax.transAxes)
    ax.text(0.75, 1, "Online\nFine Tuning",
                        ha='center', va='bottom', size=10, transform=ax.transAxes)
    return ax

def apply_ewm(df, weight):
    datas = []
    for d in df.partition_by("N data", "Algo"):
        d = d.with_columns(d["Return"].ewm_mean(alpha=weight))
        datas.append(d)
    return pl.concat(datas)

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")

    tolerance_options = ["t0-05_nd5","t0_nd1", "t0_nd5", "t0_nd10",
                         "t0-05_nd1", "t0-05_nd10",
                         "t0-1_nd1", "t0-1_nd5", "t0-1_nd10"]
    #for i in range(len(tolerance_options)):
        #tolerance_options[i] = "1000000_"+tolerance_options[i]

    folder_stub = "offline_samp"
    folder_stub_noat = "offline_samp"
    env = "antmaze"

    for t in tolerance_options:
        print(t)
        folders = [f"results/{folder_stub}/jsrlgs_at/*{t}*_{env}*.txt",
                   f"results/{folder_stub}/jsrlgs_at_nooffline/*{t}*_{env}*.txt"]
        at_folders = [f"results/{folder_stub}/jsrlgs_at/agent_types/*{t}*_{env}*agent_types.txt",
                      f"results/{folder_stub}/jsrlgs_at_nooffline/agent_types/*{t}*_{env}*agent_types.txt"]
        noat_folders = [f"results/{folder_stub_noat}/jsrlgs/*{t}*_{env}*.txt",
                        f"results/{folder_stub_noat}/jsrlgs_nooffline/*{t}*_{env}*.txt"]
        noat_at_folders = [f"results/{folder_stub_noat}/jsrlgs/agent_types/*{t}*_{env}*agent_types.txt",
                           f"results/{folder_stub_noat}/jsrlgs_nooffline/agent_types/*{t}*_{env}*agent_types.txt"]

        eval_returns = [glob.glob(folder) for folder in folders]
        agent_types = [glob.glob(folder) for folder in at_folders]

        noat_eval_returns = [glob.glob(folder) for folder in noat_folders]
        noat_agent_types = [glob.glob(folder) for folder in noat_at_folders]

        algos = ["JSRL-GS", "JSRL-GS-Noff"]

        #assert len(algos) == len(folders), f"Num folders {len(folders)} != num algorithm names {len(algos)}"
        print("Loading data...")
        try:
            all_data = polars_read(algos, eval_returns)
            all_at_data = polars_read(algos, agent_types, at=True)
            noat_all_data = polars_read(algos, noat_eval_returns)
            noat_all_at_data = polars_read(algos, noat_agent_types, at=True)

        except ValueError:
            print(f"Folders in {folders} are empty or do not exist.")
            exit()
        all_data = apply_ewm(all_data, weight=0.2)
        noat_all_data = apply_ewm(noat_all_data, weight=0.2)
        print("Data loaded. Making plots..")

        noat_idx = 0
        at_idx = 1
        yvals = ["Return", "Agent Type"]

        fig, axs = plt.subplots(2,2, figsize=(10,5))
        #axs = [axs]

        # NO AGENT THRESH
        for i, d in enumerate([noat_all_data, noat_all_at_data]):
            #print(d)
            sns.lineplot(ax=axs[noat_idx][i], x="Time Step", y=yvals[i], hue="Algo",
                         style="N data", data=d)
            axs[noat_idx][i].legend(loc='lower left')
            add_ax_decorations(axs[noat_idx][i])

        # AGENT THRESH
        for i, d in enumerate([all_data, all_at_data]):
            sns.lineplot(ax=axs[at_idx][i], x="Time Step", y=yvals[i], hue="Algo",
                                style="N data", data=d)
            # d.filter(pl.col("N data")==1000000)
            add_ax_decorations(axs[at_idx][i])
            axs[at_idx][i].legend(loc='lower left')

        print("Plots made. Saving plots...")
        plt.tight_layout()
        #plt.savefig(f"results/{folder_stub}/{t}_noffline.png")
        #plt.savefig(f"results/{folder_stub_noat}/{t}_noffline.svg", format="svg", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

