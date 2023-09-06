from train_online import main, make_save_dir
print("imported")
algo = "jsrl"
save_dir = make_save_dir(False, "FlappyBird-v0", algo, test=False)
print("made dir")
config = {"env_name": "FlappyBird-v0",
            "num_pretraining_steps": 0,
            "max_steps": 100000,
            "eval_interval": 1000,
            "seed": 0,
            "init_dataset_size": 1000000,
            "algo": algo,
            "load_model": "saved_models/20230901-095506_s2_d1000000_t0-05_nd5_flappy_heuristic_1000000_jsrl",
            "tolerance": 0,
            "n_prev_returns": 1,
            "downloaded_dataset": f"datasets/flappy_heuristic_1000000.pkl",
            "save_dir": save_dir}
main(config)