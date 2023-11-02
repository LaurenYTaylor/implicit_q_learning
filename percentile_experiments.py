from train_online import main, make_save_dir
algo = "ft"
env_name = "antmaze-umaze-v0"
save_dir = make_save_dir(False, env_name, algo, test=True)
config = {"env_name": env_name,
            "num_pretraining_steps": 0,
            "max_steps": 1000000,
            "eval_interval": 10000,
            "seed": 0,
            "init_dataset_size": 1000000,
            "algo": algo,
            #"load_model": f"saved_models/jsrlgs/20230906-163753_s0_d1000000_t0-05_nd10_antmaze_umaze_1000000",
            "load_model": f"saved_models/jsrl/20230906-163749_s0_d1000000_t0-05_nd5_antmaze_umaze_1000000",
            "tolerance": 0.05,
            "n_prev_returns": 5,
            "curriculum_stages": 10,
            "at_thresholds": False,
            "downloaded_dataset": f"datasets/antmaze_umaze_{1000000}.pkl",
            "save_dir": save_dir}
main(config)