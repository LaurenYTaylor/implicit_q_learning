from train_online import main, make_save_dir
algo = "jsrl"
env_name = "antmaze-umaze-v0"
save_dir = make_save_dir(False, env_name, algo, test=True)
config = {"env_name": env_name,
            "num_pretraining_steps": 100,
            "max_steps": 1000,
            "eval_interval": 100,
            "seed": 0,
            "init_dataset_size": 1000000,
            "algo": algo,
            #"load_model": f"saved_models/{algo}/20230906-163753_s0_d1000000_t0_nd1_antmaze_umaze_1000000",
            "tolerance": 0,
            "n_prev_returns": 1,
            "curriculum_stages": 10,
            "at_thresholds": False,
            "downloaded_dataset": f"datasets/antmaze_umaze_{1000000}.pkl",
            "save_dir": save_dir}
main(config)