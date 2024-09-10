# 1. create a virtual environment
# sudo apt-get/brew install python3-venv
# python3 -m venv reinforcement_learning_env
# source reinforcement_learning_env/bin/activate

# 2. install the required packages
# pip install stable_baselines3 "gymansium[all]" numpy

# 3. run the code
# python tai.py

import AsteroidEnv


import gymnasium as gym
from stable_baselines3 import DQN, A2C, DDPG
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(render_mode=None, continuous=True):
    env = AsteroidEnv.AsteroidDodger(render_mode=render_mode, continuous=continuous)
    return env


def visualize_random():
    env = make_env(render_mode="human", continuous=False)

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


def train_model_dqn():
    env = make_env(continuous=False)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logdir",
        # exploration_fraction=0.5,
        # exploration_final_eps=0.05,
    )

    model.learn(total_timesteps=1000000)

    env = make_env(render_mode="human", continuous=False)

    input("Press Enter to continue...")

    evaluate_policy(model, env, n_eval_episodes=3, render=True)
    env.close()

    model.save("DQN_model")


def train_model_a2c():
    env = make_env()
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./logdir")

    model.learn(total_timesteps=500000)

    input("Press Enter to continue...")

    env = make_env(render_mode="human")

    evaluate_policy(model, env, n_eval_episodes=3, render=True)
    env.close()

    model.save("A2C_model")


def train_model_ddpg():
    env = make_env()
    model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./logdir")

    model.learn(total_timesteps=150000)

    input("Press Enter to continue...")

    env = make_env(render_mode="human")

    evaluate_policy(model, env, n_eval_episodes=3, render=True)
    env.close()

    model.save("DDPG_model")


# visualize_random()
# train_model_a2c()
# train_model_ddpg()
train_model_dqn()
