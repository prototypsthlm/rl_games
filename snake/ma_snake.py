import random
import sys
from enum import Enum

import pygame


class SnakeAction(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    NOOP = 4


class SnakeGame:

    def __init__(self, grid_rows=30, grid_cols=30, fps=60):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.snake_body = [[5, 5]]
        self.fps = fps
        self.reset()
        self.current_direction = SnakeAction.MOVE_DOWN
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
        self.snake_body = [[5, 5]]
        random.seed(seed)
        self.target_position = [
            random.randint(1, self.grid_rows - 1),
            random.randint(1, self.grid_cols - 1),
        ]

    def _check_collision(self):
        if (
            self.snake_body[0][0] < 0
            or self.snake_body[0][0] >= self.grid_rows
            or self.snake_body[0][1] < 0
            or self.snake_body[0][1] >= self.grid_cols
        ):
            return True

        if self.snake_body[0] in self.snake_body[1:]:
            return True

        return False

    def _checkFoundTarget(self):
        if self.snake_body[0] == self.target_position:
            self.snake_body.append(self.snake_body[-1])
            self.target_position = [
                random.randint(1, self.grid_rows - 1),
                random.randint(1, self.grid_cols - 1),
            ]
            return True
        return False

    def _move_snake(self, new_head):
        self.snake_body.insert(0, new_head)
        found_target = self._checkFoundTarget()
        collided = self._check_collision()
        if not found_target:
            self.snake_body.pop()
        return collided, found_target

    def perform_action(self, action) -> tuple[bool, bool]:
        if action == SnakeAction.NOOP:
            return False, False

        if (
            action == SnakeAction.MOVE_UP
            and self.current_direction != SnakeAction.MOVE_DOWN
        ):
            new_head = [self.snake_body[0][0] - 1, self.snake_body[0][1]]
            collided, found_target = self._move_snake(new_head)
            self.current_direction = SnakeAction.MOVE_UP
            return collided, found_target

        elif (
            action == SnakeAction.MOVE_DOWN
            and self.current_direction != SnakeAction.MOVE_UP
        ):
            new_head = [self.snake_body[0][0] + 1, self.snake_body[0][1]]
            collided, found_target = self._move_snake(new_head)
            self.current_direction = SnakeAction.MOVE_DOWN
            return collided, found_target

        elif (
            action == SnakeAction.MOVE_LEFT
            and self.current_direction != SnakeAction.MOVE_RIGHT
        ):
            new_head = [self.snake_body[0][0], self.snake_body[0][1] - 1]
            collided, found_target = self._move_snake(new_head)
            self.current_direction = SnakeAction.MOVE_LEFT
            return collided, found_target

        elif (
            action == SnakeAction.MOVE_RIGHT
            and self.current_direction != SnakeAction.MOVE_LEFT
        ):
            new_head = [self.snake_body[0][0], self.snake_body[0][1] + 1]
            collided, found_target = self._move_snake(new_head)
            self.current_direction = SnakeAction.MOVE_RIGHT
            return collided, found_target

        return False, False

    def render(self):
        self.window.fill((0, 0, 0))
        for segment in self.snake_body:
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
                    if self.current_direction != SnakeAction.MOVE_DOWN:
                        self.current_direction = SnakeAction.MOVE_UP
                elif event.key == pygame.K_DOWN:
                    if self.current_direction != SnakeAction.MOVE_UP:
                        self.current_direction = SnakeAction.MOVE_DOWN
                elif event.key == pygame.K_LEFT:
                    if self.current_direction != SnakeAction.MOVE_RIGHT:
                        self.current_direction = SnakeAction.MOVE_LEFT
                elif event.key == pygame.K_RIGHT:
                    if self.current_direction != SnakeAction.MOVE_LEFT:
                        self.current_direction = SnakeAction.MOVE_RIGHT
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        return False


def main():
    game = SnakeGame(grid_rows=30, grid_cols=30, fps=16)
    running = True

    while running:
        running = not game._process_events()
        game.perform_action(game.current_direction)
        game.render()


if __name__ == "__main__":
    main()
