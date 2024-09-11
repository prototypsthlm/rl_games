import functools

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.utils import agent_selector, wrappers

import ma_snake as snake

NUM_ITERS = 100


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = MultiAgentSnakeEnv(render_mode=internal_render_mode)

    env = wrappers.AssertOutOfBoundsWrapper(env)

    env = wrappers.OrderEnforcingWrapper(env)
    return env


class MultiAgentSnakeEnv(ParallelEnv):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"name": "ma_snake_v0", "render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, grid_rows=30, grid_cols=30, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.game = snake.SnakeGame(
            grid_rows=grid_rows, grid_cols=grid_cols, fps=16, render_mode=render_mode
        )

        self.possible_agents = ["player_" + str(r) for r in range(2)]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        max_snake_length = self.grid_rows * self.grid_cols

        high = np.array(
            [self.grid_rows - 1, self.grid_cols - 1] * max_snake_length
            + [self.grid_rows - 1, self.grid_cols - 1]
        )

        obs_space = spaces.Box(
            low=0,
            high=high,
            shape=(2 * max_snake_length + 2,),
            dtype=np.int32,
        )
        return obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(snake.SnakeAction))

    def observe(self, agent):
        return np.array(self.observations[agent])

    def _get_obs(self, agent):
        snake_body = self.game.snakes[agent]
        target_position = self.game.target_position

        flattened_snake_body = [coord for segment in snake_body for coord in segment]
        obs = flattened_snake_body

        # Pad the observation with zeros if necessary
        max_length = 2 * self.grid_rows * self.grid_cols
        obs = obs + [0] * (max_length - len(obs))

        # Add target_position at the end
        obs = obs + list(target_position)

        return np.array(obs, dtype=np.int32)

    def _getAgentIndexFromName(self, agent):
        return int(agent.split("_")[1])

    def reset(self, seed=None, options=None):
        self.game.reset(seed=seed)
        self.agents = self.possible_agents[:]
        self.timestep = 0
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        observations = {
            agent: self._get_obs(self._getAgentIndexFromName(agent))
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        return observations, infos

    def step(self, actions):
        player_one_action = actions["player_0"]
        player_two_action = actions["player_1"]
        truncations = {agent: False for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}

        p1_collided, p2_target_reached = self.game.perform_action(
            self._getAgentIndexFromName("player_0"),
            snake.SnakeAction(player_one_action),
        )

        p2_collided, p1_target_reached = self.game.perform_action(
            self._getAgentIndexFromName("player_1"),
            snake.SnakeAction(player_two_action),
        )

        if p1_collided or p2_collided:
            terminations["player_0"] = True
            terminations["player_1"] = True
            rewards["player_0"] = -1
            rewards["player_1"] = -1

        if p1_target_reached:
            rewards["player_0"] = 1

        if p2_target_reached:
            rewards["player_1"] = 1

        # Check truncation conditions (e.g. max iterations)
        truncations = {agent: False for agent in self.agents}
        if self.timestep >= NUM_ITERS:
            rewards = {agent: 0 for agent in self.agents}
            truncations = {agent: True for agent in self.agents}
        self.timestep += 1

        observations = {
            agent: self._get_obs(self._getAgentIndexFromName(agent))
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}

        # Dont know what this does.
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.game.render()


if __name__ == "__main__":
    env = MultiAgentSnakeEnv()
    parallel_api_test(env, num_cycles=1_000_000)
