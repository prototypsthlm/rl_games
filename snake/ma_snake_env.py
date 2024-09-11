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


class SnakeGameEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 16}

    def __init__(self, grid_rows=30, grid_cols=30, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.total_steps = 0
        self.game = snake.SnakeGame(
            grid_rows=grid_rows, grid_cols=grid_cols, fps=16, render_mode=render_mode
        )

        self.action_space = spaces.Discrete(len(snake.SnakeAction))

        max_snake_length = self.grid_rows * self.grid_cols

        self.observation_space = spaces.Box(
            low=0,
            high=max(self.grid_rows, self.grid_cols),
            shape=(7,),
            dtype=np.int32,
        )

    def _get_obs(self):
        snake_body = self.game.snake_body
        snake_head_position = snake_body[0]
        target_position = self.game.target_position

        dirs_to_check = []
        curr_dir = self.game.current_direction
        if curr_dir == snake.SnakeDirection.UP:
            dirs_to_check = [[-1, 0], [0, -1], [0, 1]]
        elif curr_dir == snake.SnakeDirection.DOWN:
            dirs_to_check = [[1, 0], [0, -1], [0, 1]]
        elif curr_dir == snake.SnakeDirection.LEFT:
            dirs_to_check = [[0, -1], [-1, 0], [1, 0]]
        elif curr_dir == snake.SnakeDirection.RIGHT:
            dirs_to_check = [[0, 1], [-1, 0], [1, 0]]

        def pos_plus_movement(pos, movement):
            return [pos[0] + movement[0], pos[1] + movement[1]]

        def is_coord_free(coord):
            is_free = (
                coord not in snake_body
                and coord[0] >= 0
                and coord[0] < self.grid_rows
                and coord[1] >= 0
                and coord[1] < self.grid_cols
            )
            return max(self.grid_rows, self.grid_cols) if is_free else 0

        return np.array(
            [
                snake_head_position[0],
                snake_head_position[1],
                target_position[0],
                target_position[1],
                is_coord_free(pos_plus_movement(snake_head_position, dirs_to_check[0])),
                is_coord_free(pos_plus_movement(snake_head_position, dirs_to_check[1])),
                is_coord_free(pos_plus_movement(snake_head_position, dirs_to_check[2])),
            ],
            dtype=np.int32,
        )

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

        self.total_steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        info_before = self._get_info()
        prev_distance = info_before["snake_to_target_distance"]
        collided, target_reached = self.game.perform_action(snake.SnakeAction(action))
        info_after = self._get_info()
        new_distance = info_after["snake_to_target_distance"]

        reward = 0
        done = False
        self.total_steps += 1

        if new_distance >= prev_distance:
            reward = 1 / new_distance * -10

        if new_distance < prev_distance:
            reward = 1 / new_distance * 100

        if target_reached:
            reward = 50

        if collided:
            reward = -100
            done = True

        if self.total_steps >= 1000:
            done = True

        observation = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, info_after

    def render(self):
        return self.game.render()


def visualize_random():
    env = SnakeGameEnv(grid_rows=30, grid_cols=30, render_mode="human")
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


if __name__ == "__main__":
    visualize_random()
