import random
import sys
from enum import Enum

import pygame


class SnakeAction(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    NOOP = 5


class SnakeGame:

    def __init__(self, grid_rows=30, grid_cols=30, fps=60, move_delay=10):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.fps = fps
        self.move_delay = move_delay
        self.move_counter = 0
        self.reset()
        self.last_action = SnakeAction.NOOP
        self._init_pygame()

    def _init_pygame(self):
        pygame.display.init()

        # Clock
        self.clock = pygame.time.Clock()

        # Rendering
        self.cell_height = 16
        self.cell_width = 16

        self.window = pygame.display.set_mode(
            (self.grid_cols * self.cell_width, self.grid_rows * self.cell_height)
        )

    def reset(self, seed=None):
        self.snake_position = [0, 0]
        random.seed(seed)
        self.target_position = [
            random.randint(1, self.grid_rows - 1),
            random.randint(1, self.grid_cols - 1),
        ]

    def perform_action(self, action) -> bool:
        if self.move_counter < self.move_delay:
            self.move_counter += 1
            return 0

        self.move_counter = 0

        if action == SnakeAction.MOVE_UP:
            self.snake_position[0] = max(0, self.snake_position[0] - 1)
        elif action == SnakeAction.MOVE_DOWN:
            self.snake_position[0] = min(self.grid_rows - 1, self.snake_position[0] + 1)
        elif action == SnakeAction.MOVE_LEFT:
            self.snake_position[1] = max(0, self.snake_position[1] - 1)
        elif action == SnakeAction.MOVE_RIGHT:
            self.snake_position[1] = min(self.grid_cols - 1, self.snake_position[1] + 1)

        if self.snake_position == self.target_position:
            print("Target reached")
            return True

        return False

    def render(self):
        self.window.fill((0, 0, 0))
        pygame.draw.rect(
            self.window,
            (0, 255, 0),
            pygame.Rect(
                self.snake_position[1] * self.cell_width,
                self.snake_position[0] * self.cell_height,
                self.cell_width,
                self.cell_height,
            ),
        )
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
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.last_action = SnakeAction.MOVE_UP
                elif event.key == pygame.K_DOWN:
                    self.last_action = SnakeAction.MOVE_DOWN
                elif event.key == pygame.K_LEFT:
                    self.last_action = SnakeAction.MOVE_LEFT
                elif event.key == pygame.K_RIGHT:
                    self.last_action = SnakeAction.MOVE_RIGHT
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        return False


def main():
    game = SnakeGame(grid_rows=30, grid_cols=30, fps=60, move_delay=10)
    running = True

    while running:
        running = not game._process_events()
        game.perform_action(game.last_action)
        game.render()


if __name__ == "__main__":
    main()
