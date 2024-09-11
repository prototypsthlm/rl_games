import random
import sys
from enum import Enum

import pygame


class SnakeAction(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3


class SnakeGame:

    def __init__(self, grid_rows=30, grid_cols=30, n_players=2, fps=60, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.n_players = n_players
        self._set_random_snake_positions()
        self.fps = fps
        self.reset()
        self.snake_directions = [SnakeAction.MOVE_DOWN, SnakeAction.MOVE_DOWN]
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
        self.snakes = [[[
            random.randint(1, self.grid_rows - 1),
            random.randint(1, self.grid_cols - 1),
            ]] for _ in range(self.n_players)]

    def _set_random_target_position(self):
        self.target_position = [
            random.randint(1, self.grid_rows - 1),
            random.randint(1, self.grid_cols - 1),
        ]
    
    def reset(self, seed=None):
        random.seed(seed)
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

        return False

    def _checkFoundTarget(self, snake):
        if snake[0] == self.target_position:
            snake.append(snake[-1])
            self._set_random_target_position()
            return True
        return False

    def _move_snake(self, snake, new_head):
        snake.insert(0, new_head)
        found_target = self._checkFoundTarget(snake)
        collided = self._check_collision(snake)
        if not found_target:
            snake.pop()
        return collided, found_target

    def _is_valid_action(self, action):
        if (
            action == SnakeAction.MOVE_UP
            and self.current_direction == SnakeAction.MOVE_DOWN
        ):
            return False
        if (
            action == SnakeAction.MOVE_DOWN
            and self.current_direction == SnakeAction.MOVE_UP
        ):
            return False
        if (
            action == SnakeAction.MOVE_LEFT
            and self.current_direction == SnakeAction.MOVE_RIGHT
        ):
            return False
        if (
            action == SnakeAction.MOVE_RIGHT
            and self.current_direction == SnakeAction.MOVE_LEFT
        ):
            return False
        return True

    def perform_action(self, snake_index, action) -> tuple[bool, bool]:
        snake_body = self.snakes[snake_index]

        if action == SnakeAction.MOVE_UP:
            new_head = [snake_body[0][0] - 1, snake_body[0][1]]
            collided, found_target = self._move_snake(snake_body, new_head)
            self.snake_directions[snake_index] = SnakeAction.MOVE_UP
            return collided, found_target

        elif action == SnakeAction.MOVE_DOWN:
            new_head = [snake_body[0][0] + 1, snake_body[0][1]]
            collided, found_target = self._move_snake(snake_body, new_head)
            self.snake_directions[snake_index] = SnakeAction.MOVE_DOWN
            return collided, found_target

        elif action == SnakeAction.MOVE_LEFT:
            new_head = [snake_body[0][0], snake_body[0][1] - 1]
            collided, found_target = self._move_snake(snake_body, new_head)
            self.snake_directions[snake_index] = SnakeAction.MOVE_LEFT
            return collided, found_target

        elif action == SnakeAction.MOVE_RIGHT:
            new_head = [snake_body[0][0], snake_body[0][1] + 1]
            collided, found_target = self._move_snake(snake_body, new_head)
            self.snake_directions[snake_index] = SnakeAction.MOVE_RIGHT
            return collided, found_target

        return False, False

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return

        self.window.fill((0, 0, 0))
        
        # Iterate over each snake and draw its segments
        for snake in self.snakes:
            for segment in snake:
                pygame.draw.rect(
                    self.window,
                    (0, 255, 0),
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
            (255, 0, 0),
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if self.snake_directions[0] != SnakeAction.MOVE_DOWN:
                        self.snake_directions[0] = SnakeAction.MOVE_UP
                elif event.key == pygame.K_DOWN:
                    if self.snake_directions[0] != SnakeAction.MOVE_UP:
                        self.snake_directions[0] = SnakeAction.MOVE_DOWN
                elif event.key == pygame.K_LEFT:
                    if self.snake_directions[0] != SnakeAction.MOVE_RIGHT:
                        self.snake_directions[0] = SnakeAction.MOVE_LEFT
                elif event.key == pygame.K_RIGHT:
                    if self.snake_directions[0] != SnakeAction.MOVE_LEFT:
                        self.snake_directions[0] = SnakeAction.MOVE_RIGHT
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        return False


def main():
    game = SnakeGame(grid_rows=30, grid_cols=30, fps=16, render_mode="human")
    running = True

    while running:
        running = not game._process_events()
        game.perform_action(0, game.snake_directions[0])
        game.render()


if __name__ == "__main__":
    main()
