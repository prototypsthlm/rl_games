import gymnasium as gym
import ma_snake as snake
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id="ma_snake_env",
    entry_point="ma_snake_env:SnakeGameEnv",
)

# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/


class SnakeGameEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_rows=30, grid_cols=30, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.game = snake.SnakeGame(grid_rows=grid_rows, grid_cols=grid_cols)

        self.action_space = spaces.Discrete(len(snake.SnakeAction))

        high = np.array(
            [
                self.grid_rows - 1,
                self.grid_cols - 1,
                self.grid_rows - 1,
                self.grid_cols - 1,
            ],
        )

        self.observation_space = spaces.Box(
            low=0, high=high, shape=(4,), dtype=np.int32
        )

    def _get_obs(self):
        return np.array(
            self.game.snake_body[0] + self.game.target_position, dtype=np.int32
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

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        prev_distance = self._get_info()["snake_to_target_distance"]
        collided, target_reached = self.game.perform_action(snake.SnakeAction(action))
        reward = 0
        new_distance = self._get_info()["snake_to_target_distance"]
        done = False

        if new_distance < prev_distance:
            reward = 0.1
        elif new_distance > prev_distance:
            reward = -0.1

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


visualize_random()
