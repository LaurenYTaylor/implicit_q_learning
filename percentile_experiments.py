from train_online import main, make_save_dir
algo = "jsrl"
env_name = "FlappyBird-v0"
save_dir = make_save_dir(False, env_name, algo, test=True)
config = {"env_name": env_name,
            "num_pretraining_steps": 1000000,
            "max_steps": 1000000,
            "eval_interval": 10000,
            "seed": 0,
            "init_dataset_size": 1000000,
            "algo": algo,
            #"load_model": f"saved_models/{algo}/20230906-163753_s0_d1000000_t0_nd1_antmaze_umaze_1000000",
            "tolerance": 0,
            "n_prev_returns": 5,
            "curriculum_stages": 10,
            "at_thresholds": True,
            "warm_start": True,
            #"downloaded_dataset": f"datasets/antmaze_umaze_{1000000}.pkl",
            "downloaded_dataset": f"datasets/flappy_ppo_{1000000}.pkl",
            "save_dir": save_dir}
main(config)