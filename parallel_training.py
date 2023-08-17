import ray
from train_online import main, make_save_dir
from memory_profiler import profile

@ray.remote
def run_training(seed, n_data, algo, save_dir, config):
    print(config)
    config["seed"] = seed
    config["init_dataset_size"] = n_data
    config["save_dir"] = save_dir
    config["downloaded_dataset"] = f"datasets/antmaze_umaze_{n_data}.pkl"
    config["algo"] = algo
    return main(config)


def run(seeds, data_sizes, algo, config):
    if config["max_steps"] <= 100:
        test = True
    else:
        test = False
    save_dir = make_save_dir(False, "antmaze-umaze-v0", algo, test=test)
    object_references = [
        run_training.remote(seeds[i], data_sizes[j], algo, save_dir, config) for i in range(len(seeds)) for j in
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

    config = {"env_name": "antmaze-umaze-v0",
              "num_pretraining_steps": 1000000,
              "max_steps": 1000000}

    if args.test:
        seeds = [0]
        data_sizes = [1000000]
        config["num_pretraining_steps"] = 100
        config["max_steps"] = 100
        num_cpus = 1
    else:
        seeds = list(range(20))
        data_sizes = [1000, 10000, 100000, 1000000]
        num_cpus = 80

    algos = ["jsrlgs"]

    ray.init(num_cpus=num_cpus)
    for algo in algos:
        run(seeds, data_sizes, algo, config)
