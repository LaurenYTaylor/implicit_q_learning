import ray
from train_online import main, make_save_dir
from memory_profiler import profile

@ray.remote
def run_training(seed, n_data, algo, save_dir, config):
    config["seed"] = seed
    config["init_dataset_size"] = 1000000
    config["save_dir"] = save_dir
    config["downloaded_dataset"] = n_data
    config["algo"] = algo
    return main(config)


def run(seeds, data_sizes, algos, config):
    if config["max_steps"] <= 100:
        test = True
    else:
        test = False
    save_dirs = []
    for algo in algos:
        save_dirs.append(make_save_dir(False, config["env_name"], algo, test=False))
    object_references = [
        run_training.remote(seed, data_size, algo, save_dirs[i], config) for i, algo in enumerate(algos) for seed in
        seeds for data_size in data_sizes
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

    config = {"env_name": "FlappyBird-v0",
              "num_pretraining_steps": 1000000,
              "max_steps": 1000000}

    if args.test:
        seeds = [0]
        data_sizes = [f"datasets/flappy_heuristic_1000000.pkl", f"datasets/flappy_1000000.pkl"]
        config["num_pretraining_steps"] = 100
        config["max_steps"] = 100
        #config["eval_interval"] = 1000
        num_cpus = 1
    else:
        seeds = list(range(20))
        data_sizes = [f"datasets/flappy_heuristic_1000000.pkl", f"datasets/flappy_1000000.pkl"]
        num_cpus = 80

    algos = ["ft", "jsrl", "jsrlgs"]

    ray.init(num_cpus=num_cpus)

    run(seeds, data_sizes, algos, config)
