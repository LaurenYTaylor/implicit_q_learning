import time
import flappy_bird_gym
import pickle
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback

mode = "run"
model = "logs/sb3/best_models/best_model"
render = False

env = flappy_bird_gym.make("FlappyBird-v0")

if mode == "train":
        eval_env = flappy_bird_gym.make("FlappyBird-v0")
        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/sb3/best_models",
                                     log_path="./logs/sb3", eval_freq=10000,
                                     deterministic=True, render=False)
        if model == "PPO":
            alg = PPO("MlpPolicy", env=env, tensorboard_log="logs/sb3", verbose=1, learning_rate=1e-3, gamma=0.99,
                      n_steps=4096)
        else:
            alg = A2C("MlpPolicy", env=env, tensorboard_log="logs/sb3", verbose=1, learning_rate=1e-3)
        alg.learn(total_timesteps=int(1e7), callback=eval_callback)
elif mode == "run":
    obs = env.reset()
    data = {"observations": [], "actions": [], "rewards": [], "next_observations": [], "rewards": [], "terminals": []}
    len_data = 1000000
    if "best_model" in model:
        agent = PPO.load(model)
    i = 0
    while i < len_data:
        print(f"{i}/{len_data}")
        if model == "random":
            action = env.action_space.sample() #for a random action
        elif model == "heuristic":
            if obs[0] < 0.2 and obs[1] < 0:
                action = 1
            elif obs[1] < 0.2 and obs[1] > 0:
                action = 0
            else:
                action = env.action_space.sample()
        elif "best_model" in model:
            action = agent.predict(obs)[0]

        # Processing:
        next_obs, reward, done, info = env.step(action)
        data["observations"].append(obs)
        data["next_observations"].append(next_obs)
        data["actions"].append([action])
        data["rewards"].append(reward)
        data["terminals"].append(done)
        if render:
            env.render()
            time.sleep(1 / 30)  # FPS
    
        obs = next_obs
        if done:
            obs = env.reset()
        i += 1
    
    env.close()


for k, v in data.items():
    data[k] = np.array(v)
    print(len(v))
with open(f"datasets/flappy_ppo_{len_data}.pkl", "wb") as f:
    pickle.dump(data, f)

