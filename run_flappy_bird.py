import time
import flappy_bird_gym
import pickle
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

env = flappy_bird_gym.make("FlappyBird-v0")

#ppo = PPO("MlpPolicy", env=env, tensorboard_log="logs/sb3")

#ppo.learn(total_timesteps=int(1e6))

obs = env.reset()
data = {"observations": [], "actions": [], "rewards": [], "next_observations": [], "rewards": [], "terminals": []}

len_data = 1000000
i = 0
while i < len_data:
    # Next action:
    # (feed the observation to your agent here)
    #action = env.action_space.sample() #for a random action

    if obs[0] < 0.2 and obs[1] < 0:
        action = 1
    elif obs[1] < 0.2 and obs[1] > 0:
        action = 0
    else:
        action = env.action_space.sample()

    # Processing:
    next_obs, reward, done, info = env.step(action)
    data["observations"].append(obs)
    data["next_observations"].append(next_obs)
    data["actions"].append([action])
    data["rewards"].append(reward)
    data["terminals"].append(done)

    #env.render()
    #time.sleep(1 / 30)  # FPS

    obs = next_obs
    if done:
        obs = env.reset()
    i += 1

env.close()


for k, v in data.items():
    data[k] = np.array(v)
    print(len(v))
with open(f"datasets/flappy_heuristic_{len_data}.pkl", "wb") as f:
    pickle.dump(data, f)

