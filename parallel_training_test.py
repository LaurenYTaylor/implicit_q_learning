import ray
from train_online import main, make_save_dir
from memory_profiler import profile

@ray.remote
def run_training(seed, n_data, save_dir, config):
    config["seed"] = seed
    config["init_dataset_size"] = n_data
    config["save_dir"] = save_dir
    config["downloaded_dataset"] = f"datasets/antmaze_umaze_{n_data}.pkl"
    return main(config)


def run(seeds, data_sizes, config):
    if config["max_steps"] <= 1000:
        test = True
    else:
        test = False
    save_dir = make_save_dir(False, "antmaze-umaze-v0", config["algo"], test=test)
    object_references = [
        run_training.remote(seed, data_size, save_dir, config)
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
    ray.init(num_cpus=1)
    for algo in ["ft", "jsrl", "jsrlgs"]:
        config = {"env_name": "antmaze-umaze-v0",
	          "num_pretraining_steps": 1000,
	          "max_steps": 1000,
                  "eval_interval": 250,
	          "algo": algo}
       	seeds = [0]
        data_sizes = [1000]
        run(seeds, data_sizes, config)
