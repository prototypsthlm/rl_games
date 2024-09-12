import random
import sys
from enum import Enum, IntEnum

import pygame


class SnakeAction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SnakeGame:

    def __init__(
        self, grid_rows=30, grid_cols=30, n_players=2, fps=4, render_mode=None
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.n_players = n_players
        self._set_random_snake_positions()
        self.fps = fps
        self.frame = 0
        self.reset()
        self.snake_directions = [SnakeAction.DOWN for _ in range(n_players)]
        if self.render_mode in ["human", "rgb_array"]:
            self._init_pygame()

    def _init_pygame(self):
        print("Render mode: ", self.render_mode)
        print("Initializing Pygame")
        pygame.init()
        pygame.display.init()

        # Clock
        self.clock = pygame.time.Clock()

        # Rendering
        self.cell_height = 16
        self.cell_width = 16

        self.window = pygame.display.set_mode(
            (self.grid_cols * self.cell_width, self.grid_rows * self.cell_height)
        )

    ## Game Init Functions
    def _set_random_snake_positions(self):
        self.snakes = [
            [
                [
                    random.randint(1, self.grid_rows - 1),
                    random.randint(1, self.grid_cols - 1),
                ]
            ]
            for _ in range(self.n_players)
        ]

    def _set_random_target_position(self):
        self.target_position = [
            random.randint(1, self.grid_rows - 1),
            random.randint(1, self.grid_cols - 1),
        ]

    def reset(self, seed=None):
        random.seed(seed)
        self.frame = 0
        self.snake_directions = [SnakeAction.DOWN for _ in range(self.n_players)]
        self._set_random_snake_positions()
        self._set_random_target_position()

    def _check_collision(self, snake):
        if (
            snake[0][0] < 0
            or snake[0][0] >= self.grid_rows
            or snake[0][1] < 0
            or snake[0][1] >= self.grid_cols
        ):
            return True

        if snake[0] in snake[1:]:
            return True

        other_snake = self.snakes[1 - self.snakes.index(snake)]

        if snake[0] in other_snake:
            return True

        return False

    def _checkFoundTarget(self, snake):
        if snake[0] == self.target_position:
            snake.append(snake[-1])
            self._set_random_target_position()
            return True
        return False

    def _move_snake(self, snake_index, new_head):
        snake = self.snakes[snake_index]
        snake.insert(0, new_head)
        found_target = self._checkFoundTarget(snake)
        collided = self._check_collision(snake)
        if not found_target:
            snake.pop()
        return collided, found_target

    def _is_valid_action(self, direction, action):
        if action == SnakeAction.UP and direction == SnakeAction.DOWN:
            return False
        if action == SnakeAction.DOWN and direction == SnakeAction.UP:
            return False
        if action == SnakeAction.LEFT and direction == SnakeAction.RIGHT:
            return False
        if action == SnakeAction.RIGHT and direction == SnakeAction.LEFT:
            return False
        return True

    def perform_action(self, snake_index, action) -> tuple[bool, bool]:
        snake_body = self.snakes[snake_index]
        snake_direction = self.snake_directions[snake_index]
        if not self._is_valid_action(snake_direction, action):
            action = snake_direction

        if action == SnakeAction.UP:
            new_head = [snake_body[0][0] - 1, snake_body[0][1]]
            collided, found_target = self._move_snake(snake_index, new_head)
            self.snake_directions[snake_index] = SnakeAction.UP
            return collided, found_target

        elif action == SnakeAction.DOWN:
            new_head = [snake_body[0][0] + 1, snake_body[0][1]]
            collided, found_target = self._move_snake(snake_index, new_head)
            self.snake_directions[snake_index] = SnakeAction.DOWN
            return collided, found_target

        elif action == SnakeAction.LEFT:
            new_head = [snake_body[0][0], snake_body[0][1] - 1]
            collided, found_target = self._move_snake(snake_index, new_head)
            self.snake_directions[snake_index] = SnakeAction.LEFT
            return collided, found_target

        elif action == SnakeAction.RIGHT:
            new_head = [snake_body[0][0], snake_body[0][1] + 1]
            collided, found_target = self._move_snake(snake_index, new_head)
            self.snake_directions[snake_index] = SnakeAction.RIGHT
            return collided, found_target

        return False, False

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return

        self.window.fill((0, 0, 0))

        # Iterate over each snake and draw its segments
        print("Rendering frame: ", self.frame)
        print("Render mode: ", self.render_mode)
        self.frame += 1
        for index, snake in enumerate(self.snakes):
            for segment in snake:
                pygame.draw.rect(
                    self.window,
                    (0, 255, 0) if index == 0 else (0, 0, 255),
                    pygame.Rect(
                        segment[1] * self.cell_width,
                        segment[0] * self.cell_height,
                        self.cell_width,
                        self.cell_height,
                    ),
                )

        # Draw the target
        pygame.draw.rect(
            self.window,
            (255, 0, 0) if index == 0 else (0, 255, 255),
            pygame.Rect(
                self.target_position[1] * self.cell_width,
                self.target_position[0] * self.cell_height,
                self.cell_width,
                self.cell_height,
            ),
        )

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.fps)

        if self.render_mode == "rgb_array":
            return self._get_rgb_array()

    def _get_rgb_array(self):
        """Capture the current screen as an RGB array."""
        return pygame.surfarray.array3d(pygame.display.get_surface())

    def _process_events(self):
        action = self.snake_directions[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = SnakeAction.UP
                elif event.key == pygame.K_DOWN:
                    action = SnakeAction.DOWN
                elif event.key == pygame.K_LEFT:
                    action = SnakeAction.LEFT
                elif event.key == pygame.K_RIGHT:
                    action = SnakeAction.RIGHT
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        return True, action


def main():
    game = SnakeGame(grid_rows=30, grid_cols=30, fps=4, render_mode="human")
    running = True

    while running:
        running, action = game._process_events()
        game.perform_action(0, action)
        game.render()


if __name__ == "__main__":
    main()
