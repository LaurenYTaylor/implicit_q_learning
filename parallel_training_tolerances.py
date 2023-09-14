import os
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

import ray
from train_online import main, make_save_dir
from memory_profiler import profile
import time

@ray.remote
#@profile
def run_training(seed, n_data, tolerance, n_prev, algo, save_dir, config):
    config["seed"] = seed
    config["init_dataset_size"] = n_data
    config["save_dir"] = save_dir
    config["downloaded_dataset"] = f"datasets/antmaze_umaze_{n_data}.pkl"
    config["algo"] = algo
    config["tolerance"] = tolerance
    config["n_prev_returns"] = n_prev
    return main(config)


def run(seeds, data_sizes, algos, config, tolerances, n_prevs):
    if config["max_steps"] <= 1000:
        test = True
    else:
        test = False
    save_dirs = []
    for a in algos:
        save_dirs.append(make_save_dir(False, "antmaze-umaze-v0", a, test=test))
    object_references = [
        run_training.remote(s, d, t, n, a, save_dirs[i], config) for s in seeds for d in
        data_sizes for t in tolerances for n in n_prevs for i, a in enumerate(algos)
    ]

    all_data = []
    while len(object_references) > 0:
        finished, object_references = ray.wait(
            object_references, timeout=2.0
        )
        data = ray.get(finished)
        all_data.extend(data)


if __name__ == "__main__":
    start = time.time()
    print(f"Start: {start}")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    config = {"env_name": "antmaze-umaze-v0",
              "num_pretraining_steps": 1000000,
              "max_steps": 1000000}

    if args.test:
        seeds = list(range(1))
        data_sizes = [1000, 10000, 100000, 1000000]
        config["num_pretraining_steps"] = 1000
        config["max_steps"] = 1000
        config["eval_interval"] = 100
        num_cpus = 1
    else:
        seeds = [0]
        #data_sizes = [1000, 10000, 100000, 1000000]
        data_sizes = [1000000, 1000]
        num_cpus = 36
    ray.init(num_cpus=num_cpus)

    algos = ["jsrl", "jsrlgs"]
    run(seeds, data_sizes, algos, config, tolerances=[0, 0.05, 0.1], n_prevs=[1, 5, 10])
    print(f"End: {(time.time()-start)/60} min")
