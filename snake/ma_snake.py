import random
import sys
from enum import IntEnum, Enum

import pygame


class SnakeAction(Enum):
    NOTHING = 0
    RIGHT = 1
    LEFT = 2


class SnakeDirection(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SnakeGame:
    def __init__(self, grid_rows=30, grid_cols=30, fps=60, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.snake_body = [[4, 4]]
        self.fps = fps
        self.reset()
        self.current_direction = SnakeDirection.DOWN
        if self.render_mode in ["human", "rgb_array"]:
            self._init_pygame()

    def _init_pygame(self):
        print("Render mode: ", self.render_mode)
        print("Initializing Pygame")
        pygame.init()

        # Rendering
        self.cell_height = 16
        self.cell_width = 16

        if self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_cols * self.cell_width, self.grid_rows * self.cell_height)
            )
        elif self.render_mode == "rgb_array":
            self.window = pygame.Surface(
                (self.grid_cols * self.cell_width, self.grid_rows * self.cell_height)
            )

        # Clock
        self.clock = pygame.time.Clock()

    def reset(self, seed=None):
        self.snake_body = [[4, 4]]
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
        if action == SnakeAction.NOTHING:
            next_head_position_dir = [0, 0]
            if self.current_direction == SnakeDirection.UP:
                next_head_position_dir = [-1, 0]
            elif self.current_direction == SnakeDirection.DOWN:
                next_head_position_dir = [1, 0]
            elif self.current_direction == SnakeDirection.LEFT:
                next_head_position_dir = [0, -1]
            elif self.current_direction == SnakeDirection.RIGHT:
                next_head_position_dir = [0, 1]

            new_head = [
                self.snake_body[0][0] + next_head_position_dir[0],
                self.snake_body[0][1] + next_head_position_dir[1],
            ]
            collided, found_target = self._move_snake(new_head)
            return collided, found_target

        elif action == SnakeAction.LEFT:
            next_head_position_dir = [0, 0]
            new_direction = SnakeDirection.UP
            if self.current_direction == SnakeDirection.UP:
                new_direction = SnakeDirection.LEFT
                next_head_position_dir = [0, -1]
            elif self.current_direction == SnakeDirection.DOWN:
                new_direction = SnakeDirection.RIGHT
                next_head_position_dir = [0, 1]
            elif self.current_direction == SnakeDirection.LEFT:
                new_direction = SnakeDirection.DOWN
                next_head_position_dir = [1, 0]
            elif self.current_direction == SnakeDirection.RIGHT:
                new_direction = SnakeDirection.UP
                next_head_position_dir = [-1, 0]

            new_head = [
                self.snake_body[0][0] + next_head_position_dir[0],
                self.snake_body[0][1] + next_head_position_dir[1],
            ]
            collided, found_target = self._move_snake(new_head)
            self.current_direction = new_direction
            return collided, found_target

        elif action == SnakeAction.RIGHT:
            next_head_position_dir = [0, 0]
            new_direction = SnakeDirection.UP
            if self.current_direction == SnakeDirection.UP:
                new_direction = SnakeDirection.RIGHT
                next_head_position_dir = [0, 1]
            elif self.current_direction == SnakeDirection.DOWN:
                new_direction = SnakeDirection.LEFT
                next_head_position_dir = [0, -1]
            elif self.current_direction == SnakeDirection.LEFT:
                new_direction = SnakeDirection.UP
                next_head_position_dir = [-1, 0]
            elif self.current_direction == SnakeDirection.RIGHT:
                new_direction = SnakeDirection.DOWN
                next_head_position_dir = [1, 0]

            new_head = [
                self.snake_body[0][0] + next_head_position_dir[0],
                self.snake_body[0][1] + next_head_position_dir[1],
            ]
            collided, found_target = self._move_snake(new_head)
            self.current_direction = new_direction
            return collided, found_target

        return False, False

    def render(self):

        if self.render_mode not in ["human", "rgb_array"]:
            return

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
        pygame.event.pump()
        if self.render_mode == "human":
            pygame.display.flip()
        self.clock.tick(self.fps)

        if self.render_mode == "rgb_array":
            return self._get_rgb_array()

    def _get_rgb_array(self):
        """Capture the current screen as an RGB array."""
        return pygame.surfarray.array3d(self.window)

    def _process_events(self):
        action = SnakeAction.NOTHING

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = SnakeAction.LEFT
                elif event.key == pygame.K_RIGHT:
                    action = SnakeAction.RIGHT
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        return True, action


def main():
    game = SnakeGame(grid_rows=30, grid_cols=30, fps=16, render_mode="human")
    running = True

    while running:
        running, action = game._process_events()
        # input("Press Enter to continue...")
        game.perform_action(action)
        game.render()


if __name__ == "__main__":
    main()
