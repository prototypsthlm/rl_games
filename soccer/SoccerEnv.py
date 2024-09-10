import pygame
from Box2D.b2 import world, edgeShape, vec2

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle

import numpy as np

# Define screen dimensions
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 300

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (169, 169, 169)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Physics world setup
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS

PLAYER_SIZE = 20


def calc_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class SoccerBoard(gym.Env, EzPickle):
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),  # 6 fixed-size components + MAX_RAYCASTS distances
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.setup()

    def setup(self):
        self.step_count = 0
        self.terminate = False
        self.total_reward = 0
        self.player1_score = 0

        self.world = world(gravity=(0, 0), doSleep=True)

        self.ball = self.world.CreateDynamicBody(
            position=((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2)), angle=0
        )
        self.ball_shape = self.ball.CreateCircleFixture(
            radius=5,
            density=0.01,
            friction=0.01,
            restitution=1,
        )

        self.player1 = self.world.CreateDynamicBody(
            position=((SCREEN_WIDTH - 50), (SCREEN_HEIGHT / 2 - 100)), angle=0
        )

        # create rect shape
        self.player1_shape = self.player1.CreatePolygonFixture(
            box=(PLAYER_SIZE / 2, PLAYER_SIZE / 2),
            density=0.01,
            friction=0.1,
            restitution=1,
        )

        self.walls = [
            # Top wall
            self.world.CreateStaticBody(
                position=(0, 0), shapes=edgeShape(vertices=[(0, 0), (SCREEN_WIDTH, 0)])
            ),
            # Right wall
            self.world.CreateStaticBody(
                position=(SCREEN_WIDTH, 0),
                shapes=edgeShape(vertices=[(0, 0), (0, SCREEN_HEIGHT)]),
            ),
            # Bottom wall
            self.world.CreateStaticBody(
                position=(0, SCREEN_HEIGHT),
                shapes=edgeShape(vertices=[(0, 0), (SCREEN_WIDTH, 0)]),
            ),
            # Left wall
            self.world.CreateStaticBody(
                position=(0, 0), shapes=edgeShape(vertices=[(0, 0), (0, SCREEN_HEIGHT)])
            ),
        ]

    def _destroy(self):
        self.world.DestroyBody(self.ball)
        self.world.DestroyBody(self.player1)

        for wall in self.walls:
            self.world.DestroyBody(wall)

    def reset(self, seed=None, options=None):
        super().reset()
        self._destroy()

        self.setup()

        return [0, 0, 0, 0, 0, 0], None

    def step(self, action):
        movement = (0, 0)

        if action == 0:
            movement = (0, 1)
        elif action == 1:
            movement = (0, -1)
        elif action == 2:
            movement = (1, 0)
        elif action == 3:
            movement = (-1, 0)

        self.step_count += 1

        old_player1_position = (self.player1.position[0], self.player1.position[1])

        if movement != (0, 0):
            self.player1.ApplyForceToCenter(vec2(movement) * 100000, True)
        else:
            self.player1.linearVelocity *= 0.1

        self.world.Step(TIME_STEP, 6, 2)

        if self.render_mode == "human":
            self.render()

        reward = 0

        if self.step_count >= 1000:
            self.terminate = True
            reward -= 200

        distance_ball_center = self.ball.position[0] - SCREEN_WIDTH / 2
        reward += distance_ball_center * -0.0001

        old_distance = calc_distance(old_player1_position, self.ball.position)
        new_distance = calc_distance(self.player1.position, self.ball.position)

        if old_distance - new_distance > 0:
            reward += 1
        else:
            reward -= 1

        if (
            self.ball.position[0] <= 10
            and self.ball.position[1] >= SCREEN_HEIGHT / 2 - 40
            and self.ball.position[1] <= SCREEN_HEIGHT / 2 + 40
        ):
            self.player1_score += 1
            self.terminate = True
            reward += 500

        self.state = [
            self.player1.position[0],
            self.player1.position[1],
            self.ball.position[0],
            self.ball.position[1],
            self.ball.linearVelocity[0],
            self.ball.linearVelocity[1],
        ]

        self.total_reward += reward

        return (
            self.state,
            reward,
            self.terminate,
            False,
            {},
        )

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Soccer Multiagent")
            else:
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.terminate = False
                self._destroy()
                exit()

        self.screen.fill(GRAY)

        # player 1
        pygame.draw.rect(
            self.screen,
            GREEN,
            (
                int(self.player1.position[0] - PLAYER_SIZE / 2),  # Adjust for box size
                SCREEN_HEIGHT - int(self.player1.position[1] + PLAYER_SIZE / 2),
                PLAYER_SIZE,
                PLAYER_SIZE,
            ),
        )

        # ball
        pygame.draw.circle(
            self.screen,
            YELLOW,
            (
                int(self.ball.position[0]),
                SCREEN_HEIGHT - int(self.ball.position[1]),
            ),
            self.ball_shape.shape.radius,
        )

        # goal
        pygame.draw.rect(
            self.screen,
            WHITE,
            (
                0,
                SCREEN_HEIGHT / 2 - 40,
                10,
                80,
            ),
        )

        font = pygame.font.Font(None, 32)
        debug_reward = f"Reward: {self.total_reward:.2f}"
        debug_score_player_1 = f"Score: {self.player1_score} / 0"
        text_reward = font.render(debug_reward, True, WHITE)
        text_score_player_1 = font.render(debug_score_player_1, True, WHITE)
        self.screen.blit(text_reward, (0, 0))
        self.screen.blit(text_score_player_1, (200, 0))

        if self.render_mode == "human":
            self.clock.tick(TARGET_FPS)
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))


def main():
    soccer = SoccerBoard(render_mode="human")

    soccer.render()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        action = 4
        if keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_DOWN]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2
        elif keys[pygame.K_LEFT]:
            action = 3

        obs, reward, terminate, k, info = soccer.step(action)

        if terminate:
            soccer.reset()


if __name__ == "__main__":
    main()
