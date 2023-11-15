import ray
from train_online import main, make_save_dir
from memory_profiler import profile

@ray.remote
def run_training(seed, n_data, save_dir, config, dataset_name):
    config["seed"] = seed
    config["init_dataset_size"] = n_data
    config["save_dir"] = save_dir
    config["downloaded_dataset"] = f"datasets/{dataset_name}.pkl"
    print(config)
    return main(config)


def run(seeds, data_sizes, config, dataset_name, test=False):
    save_dir = make_save_dir(False, config["env_name"], config["algo"], test=test)
    object_references = [
        run_training.remote(seed, data_size, save_dir, config, dataset_name)
        for data_size in data_sizes for seed in seeds
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
    parser.add_argument("--algo", default="ft")
    parser.add_argument("--at_thresh", action="store_true")
    parser.add_argument("--env_name", default="antmaze-umaze-v2")
    parser.add_argument("--dataset_name", default="antmaze_umaze-v2")
    parser.add_argument("--warm_start", action="store_true")
    args = parser.parse_args()

    config = {"env_name": args.env_name,
              "num_pretraining_steps": 1000000,
              "max_steps": 1000000,
              "algo": args.algo,
              "at_thresholds": args.at_thresh,
              "eval_episodes": 100,
              "tolerance": 0,
              "n_prev_returns": 5,
              "warm_start": args.warm_start}

    if args.test:
        seeds = [0]
        data_sizes = [1000000]
        config["num_pretraining_steps"] = 50000
        config["max_steps"] = 10000
        config["eval_interval"] = 10000
        num_cpus = 1
    else:
        seeds = list(range(20))
        if "Flappy" in args.env_name:
            data_sizes = [1000000]
        else:
            data_sizes = [1000, 10000, 100000, 1000000]
        num_cpus = min(80, len(data_sizes)*len(seeds))

    ray.init(num_cpus=num_cpus)

    run(seeds, data_sizes, config, args.dataset_name, args.test)
