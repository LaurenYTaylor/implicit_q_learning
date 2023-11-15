import numpy as np
def flappy_height_goal(_env, obs):
    return np.abs(obs[1])
def flappy_all_goal(_env, obs):
    return np.sqrt(obs[0] ** 2 + obs[1] ** 2)
def antumaze_goal(env, _obs):
    return np.linalg.norm(np.array(env.target_goal) - np.array(env.get_xy()))


ENV_GOALS = {"FlappyBird-v0": flappy_height_goal,
             "antmaze-umaze-v0": antumaze_goal,
             "antmaze-umaze-diverse-v0": antumaze_goal,
             "antmaze-umaze-v2": antumaze_goal,
             "antmaze-medium-play-v2": antumaze_goal,
             "antmaze-medium-diverse-v2": antumaze_goal,
             "antmaze-large-play-v2": antumaze_goal,
             "antmaze-large-diverse-v2": antumaze_goal}