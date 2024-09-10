import argparse
import os

import gymnasium as gym
import ma_snake as snake
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from colorama import Fore, Style, init
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

init(autoreset=True)

register(
    id="ma_snake_env",
    entry_point="ma_snake_env:SnakeGameEnv",
)


class SnakeGameEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 16}

    def __init__(self, grid_rows=30, grid_cols=30, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.game = snake.SnakeGame(
            grid_rows=grid_rows, grid_cols=grid_cols, fps=16, render_mode=render_mode
        )

        self.action_space = spaces.Discrete(len(snake.SnakeAction))

        max_snake_length = self.grid_rows * self.grid_cols

        high = np.array(
            [self.grid_rows - 1, self.grid_cols - 1] * max_snake_length
            + [self.grid_rows - 1, self.grid_cols - 1]
        )

        self.observation_space = spaces.Box(
            low=0,
            high=high,
            shape=(2 * max_snake_length + 2,),
            dtype=np.int32,
        )

    def _get_obs(self):
        snake_body = self.game.snake_body
        target_position = self.game.target_position

        flattened_snake_body = [coord for segment in snake_body for coord in segment]
        obs = flattened_snake_body

        # Pad the observation with zeros if necessary
        max_length = 2 * self.grid_rows * self.grid_cols
        obs = obs + [0] * (max_length - len(obs))

        # Add target_position at the end
        obs = obs + list(target_position)

        return np.array(obs, dtype=np.int32)

    def _get_info(self):
        return {
            "snake_to_target_distance": np.linalg.norm(
                np.array(self.game.snake_body[0]) - np.array(self.game.target_position),
                ord=1,
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        prev_distance = self._get_info()["snake_to_target_distance"]
        collided, target_reached = self.game.perform_action(snake.SnakeAction(action))
        reward = -0.1
        new_distance = self._get_info()["snake_to_target_distance"]
        done = False

        if new_distance >= prev_distance:
            reward = -0.3

        if new_distance < prev_distance:
            reward = 0.1

        if target_reached:
            reward = 1

        if collided:
            reward = -1
            done = True

        info = self._get_info()
        observation = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, info

    def render(self):
        self.game.render()


def visualize_random():
    env = gym.make("ma_snake_env", render_mode="human")
    print("Checking environment")
    check_env(env.unwrapped)
    print("Environment checked")
    episodes = 5
    scores = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        score = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            print("Episode:", episode)
            print("Action:", action)
            print("Observation:", obs)
            print("Reward:", reward)
            print("Done:", done)
            print("Info:", info)
            print("Score:", score)
            score += reward
        scores.append(score)
    print("Scores:", scores)


def train_model(timesteps=250000, iters=1, replace=False):
    print("Training model for ", timesteps, " timesteps over ", iters, " iterations.")
    model_dir = "models/"
    model_name = "dqn_ma_snake"
    os.makedirs(model_dir, exist_ok=True)
    env = gym.make("ma_snake_env")

    model_path = f"{model_dir}{model_name}{timesteps*iters}"

    if replace and os.path.exists(model_path + ".zip"):
        print("Replacing existing model.")
        os.remove(model_path + ".zip")

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="tlogs/",
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
    )
    i = 0
    while i < iters:
        i += 1
        model.learn(
            total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=True
        )
        model.save(f"{model_dir}{model_name}{timesteps*i}")


def test_model(model_name: str):
    print(Fore.CYAN + "Testing model:", Fore.YELLOW + model_name)
    env = gym.make("ma_snake_env", render_mode="human")

    model = DQN.load(model_name, env=env)

    obs = env.reset()[0]
    done = False
    score = 0
    while True:
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        score += reward
        if done:
            print(Fore.GREEN + "DONE")
            print(Fore.MAGENTA + "Score:", Fore.YELLOW + str(score))
            score = 0
            env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test DQN model for Snake game."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=250000,
        help="Number of timesteps for training.",
    )
    parser.add_argument(
        "--iters", type=int, default=1, help="Number of iterations for training."
    )
    parser.add_argument(
        "-r", "--replace", action="store_true", help="Replace and overwrite the existing model."
    )
    args = parser.parse_args()

    train_model(timesteps=args.timesteps, iters=args.iters, replace=args.replace)
    test_model(f"models/dqn_ma_snake{args.timesteps * args.iters}")