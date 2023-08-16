import ray
from train_online import main, make_save_dir
from memory_profiler import profile

config = {"env_name": "antmaze-umaze-v0",
          "num_pretraining_steps": 1000000,
          "max_steps": 1000000}

@ray.remote
def run_training(seed, n_data, algo, save_dir):
    config["seed"] = seed
    config["init_dataset_size"] = n_data
    config["save_dir"] = save_dir
    config["downloaded_dataset"] = f"datasets/antmaze_umaze_{n_data}.pkl"
    config["algo"] = algo
    return main(config)


def run(seeds, data_sizes, algo, num_cpus):
    if config["max_steps"] <= 100:
        test = True
    else:
        test = False
    save_dir = make_save_dir(False, "antmaze-umaze-v0", algo, test=test)
    object_references = [
        run_training.remote(seeds[i], data_sizes[j], algo, save_dir) for i in range(len(seeds)) for j in
        range(len(data_sizes))
    ]

    all_data = []
    while len(object_references) > 0:
        finished, object_references = ray.wait(
            object_references, timeout=7.0
        )
        data = ray.get(finished)
        all_data.extend(data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        seeds = [0]
        data_sizes = [1000]
        config["num_pretraining_steps"] = 100
        config["max_steps"] = 100
        num_cpus = 1
    else:
        seeds = [0]
        data_sizes = [1000, 10000, 100000, 1000000]
        num_cpus = 72
    ray.init(num_cpus=num_cpus)

    for tolerance in [0, 0.05, 0.1]:
        for n_prev in [1, 5, 10]:
            config["tolerance"] = tolerance
            config["n_prev_returns"] = n_prev

            algos = ["jsrl", "jsrlgs"]

            for algo in algos:
                run(seeds, data_sizes, algo, num_cpus)
