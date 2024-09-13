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

    def __init__(self, grid_rows=64, grid_cols=64, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.n_players = 8
        self.game = snake.SnakeGame(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            fps=30,
            render_mode=render_mode,
            n_players=self.n_players,
        )

        self.possible_agents = ["player_" + str(r) for r in range(self.n_players)]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs_space = spaces.Box(
            low=0,
            high=max(self.grid_rows, self.grid_cols),
            shape=(12,),
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
        snake_head_position = snake_body[0]
        target_position = self.game.target_position

        def is_coord_free(coord):
            snake_neck = snake_body[1] if len(snake_body) > 1 else None

            if coord == snake_neck:
                return 0

            is_free = (
                coord not in snake_body
                and all(coord not in other_snake for other_snake in self.game.snakes)
                and coord[0] >= 0
                and coord[0] < self.grid_rows
                and coord[1] >= 0
                and coord[1] < self.grid_cols
            )
            return max(self.grid_rows, self.grid_cols) if is_free else 0

        # Collect head positions of all snakes
        head_positions = []
        for i in range(self.game.n_players):
            if i != agent:
                other_snake_body = self.game.snakes[i]
                other_snake_head_position = other_snake_body[0]
                head_positions.extend(other_snake_head_position)

        # Add the target position and free coordinates around the snake's head
        observation = np.array(
            [
                snake_head_position[0],
                snake_head_position[1],
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

        return observation

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
        truncations = {agent: False for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}

        for agent in self.agents:
            agent_index = self._getAgentIndexFromName(agent)
            action = snake.SnakeAction(actions[agent])
            collided, target_reached = self.game.perform_action(agent_index, action)

            if collided:
                rewards[agent] = -1
                terminations[agent] = True
                self._removeAgent(agent)
                # Terminate all players if one collides
                # for other_agent in self.agents:
                #     terminations[other_agent] = True

            if target_reached:
                print(f"Agent {agent} reached the target!")
                rewards[agent] = 1

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

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, new_infos

    def render(self):
        return self.game.render()

    def _removeAgent(self, agent):
        agent_index = self._getAgentIndexFromName(agent)
        self.game.dead_snakes.append(agent_index)
        self.agents.remove(agent)


if __name__ == "__main__":
    env = parallel_env()
    parallel_api_test(env, num_cycles=1_000_000)
