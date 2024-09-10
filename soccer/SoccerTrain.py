# 1. create a virtual environment
# sudo apt-get/brew install python3-venv
# python3 -m venv reinforcement_learning_env
# source reinforcement_learning_env/bin/activate

# 2. install the required packages
# pip install stable_baselines3 "gymansium[all]" numpy

# 3. run the code
# python tai.py

import SoccerEnv


import gymnasium as gym
from stable_baselines3 import DQN, A2C, DDPG, PPO
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(render_mode=None):
    env = SoccerEnv.SoccerBoard(render_mode=render_mode)
    return env


def visualize_random():
    env = make_env(render_mode="human")

    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        print(state)
        done = False
        score = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            print(action)
            obs, reward, done, something, info = env.step(action)
            score += reward
        # print("Episode:{} Score:{}".format(episode, score))
    env.close()


def train_model_ppo():
    env = make_env(render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./videos",
        name_prefix="test-video",
        episode_trigger=lambda x: x % 75 == 0 and x != 0,
    )
    env.start_video_recorder()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logdir",
        # exploration_fraction=0.05,
        # exploration_final_eps=0.05,
    )

    model.learn(total_timesteps=500000)

    env.close_video_recorder()
    del env

    env = make_env(render_mode="human")

    input("Press Enter to continue...")

    evaluate_policy(model, env, n_eval_episodes=3, render=True)
    env.close()

    model.save("PPO_model")


# visualize_random()
train_model_ppo()
