import functools

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.utils import agent_selector, parallel_to_aec, wrappers

import ma_snake as snake

NUM_ITERS = 10000


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)

    env = wrappers.AssertOutOfBoundsWrapper(env)

    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"name": "ma_snake_v0", "render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid_rows=32, grid_cols=32, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.game = snake.SnakeGame(
            grid_rows=grid_rows, grid_cols=grid_cols, fps=30, render_mode=render_mode
        )

        self.possible_agents = ["player_" + str(r) for r in range(2)]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs_space = spaces.Box(
            low=0,
            high=max(self.grid_rows, self.grid_cols),
            shape=(14,),
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
        other_snake_body = self.game.snakes[1 - agent]
        snake_head_position = snake_body[0]
        other_snake_head_position = other_snake_body[0]
        target_position = self.game.target_position

        def pos_plus_movement(pos, movement):
            return [pos[0] + movement[0], pos[1] + movement[1]]

        def is_coord_free(coord):
            snake_neck = snake_body[1] if len(snake_body) > 1 else None

            if coord == snake_neck:
                return 0

            is_free = (
                coord not in snake_body
                and coord not in other_snake_body
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
                other_snake_head_position[0],
                other_snake_head_position[1],
                target_position[0],
                target_position[1],
                is_coord_free([snake_head_position[0], snake_head_position[1] - 1]),
                is_coord_free([snake_head_position[0], snake_head_position[1] + 1]),
                is_coord_free([snake_head_position[0] + 1, snake_head_position[1]]),
                is_coord_free([snake_head_position[0] - 1, snake_head_position[1]]),
                is_coord_free([snake_head_position[0] - 1, snake_head_position[1] - 1]),
                is_coord_free([snake_head_position[0] + 1, snake_head_position[1] + 1]),
                is_coord_free([snake_head_position[0] - 1, snake_head_position[1] + 1]),
                is_coord_free([snake_head_position[0] + 1, snake_head_position[1] - 1]),
            ],
            dtype=np.int32,
        )

    def _get_info(self, agent):
        return {
            "to_target": np.linalg.norm(
                np.array(self.game.snakes[self._getAgentIndexFromName(agent)][0])
                - np.array(self.game.target_position),
                ord=1,
            )
        }

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

        # input("Press Enter to continue...")
        # print("Player 0 action: ", player_one_action)
        # print("Player 1 action: ", player_two_action)

        p1_collided, p1_target_reached = self.game.perform_action(
            self._getAgentIndexFromName("player_0"),
            snake.SnakeAction(player_one_action),
        )

        p2_collided, p2_target_reached = self.game.perform_action(
            self._getAgentIndexFromName("player_1"),
            snake.SnakeAction(player_two_action),
        )

        if p1_collided:
            rewards["player_0"] = -1
            terminations["player_0"] = True
            terminations["player_1"] = True

        if p2_collided:
            rewards["player_1"] = -1
            terminations["player_1"] = True
            terminations["player_0"] = True

        if p1_target_reached:
            rewards["player_0"] = 1

        if p2_target_reached:
            rewards["player_1"] = 1

        # input("Press Enter to continue...")
        # print(self._get_obs(0))
        # print(self._get_obs(1))

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

        new_infos = {agent: self._get_info(agent) for agent in self.agents}

        # Dont know what this does.
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, new_infos

    def render(self):
        return self.game.render()


if __name__ == "__main__":
    env = parallel_env()
    parallel_api_test(env, num_cycles=1_000_000)
