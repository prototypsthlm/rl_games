import pygame
from Box2D.b2 import world, vec2
from Box2D import b2RayCastCallback, b2Vec2
import random
import math

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle

import numpy as np

# Define screen dimensions
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700

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
SHIP_THRUST = 100  # The thrust power of the spaceship
ROTATE_SPEED = 0.1  # Rotation speed for left/right turning

NUM_ASTEROIDS = 10

LANDING_ZONE_SIZE = 150


# Rotation matrix for 2D (counterclockwise rotation)
def rotate_point(point, angle):
    x, y = point
    rotated_x = x * math.cos(angle) - y * math.sin(angle)
    rotated_y = x * math.sin(angle) + y * math.cos(angle)
    return rotated_x, rotated_y


def calculate_raycast(ship_body, angle, distance):
    # Convert ship position to Box2D coordinate system if necessary
    start_point = b2Vec2(ship_body.position[0], ship_body.position[1])

    # Calculate forward direction
    forward_direction = rotate_point((0, -distance), -angle)

    # End point of the raycast
    end_point = start_point + b2Vec2(forward_direction[0], forward_direction[1])

    return start_point, end_point


class RaycastCallback(b2RayCastCallback):
    def __init__(self):
        super().__init__()
        self.hit = False
        self.point = None
        self.normal = None
        self.fixture = None

    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True
        self.point = point
        self.normal = normal
        self.fixture = fixture
        # Returning fraction allows us to control what part of the ray we care about
        # Returning 1 continues through all fixtures, less than 1 stops early
        return fraction


class AsteroidDodger(gym.Env, EzPickle):
    def __init__(self, render_mode=None, continuous=True):
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),  # 6 fixed-size components + MAX_RAYCASTS distances
            dtype=np.float32,
        )
        self.continuous = continuous

        if continuous:
            self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.setup()

    def setup(self):
        self.raycasts = []
        self.raycast_hits = []
        self.asteroids = []
        self.thruster_active = False
        self.distance_to_landing_zone = 0
        self.step_count = 0
        self.terminate = False
        self.total_reward = 0
        self.distance_step_rewards = list(range(1350, 0, -10)) + [50]

        self.world = world(gravity=(0, 0), doSleep=True)

        self.ship_body = self.world.CreateDynamicBody(
            position=((SCREEN_WIDTH - 50), (SCREEN_HEIGHT - 50)), angle=math.pi / 4
        )
        self.ship_shape = self.ship_body.CreatePolygonFixture(
            vertices=[(-10, -10), (10, -10), (10, 10), (-10, 10)],
            density=0.01,
            friction=0.1,
        )

        for i in range(NUM_ASTEROIDS):
            offset_to_border = 25
            asteroid_body = self.world.CreateStaticBody(
                position=(
                    random.uniform(offset_to_border, SCREEN_WIDTH - offset_to_border),
                    random.uniform(offset_to_border, SCREEN_HEIGHT - offset_to_border),
                )
            )
            radius = random.randint(15, 40)
            asteroid_body.CreateCircleFixture(
                radius=radius,
                density=1,
                friction=0,
                restitution=0,
            )
            self.asteroids.append((asteroid_body, radius))

    def _destroy(self):
        self.world.DestroyBody(self.ship_body)
        for body, _ in self.asteroids:
            self.world.DestroyBody(body)

    def reset(self, seed=None):
        super().reset()
        self._destroy()

        self.setup()

        self.calculate_state([0, 0])

        return np.array(self.state, dtype=np.float32), {}

    def perform_raycast(self, start_point, end_point):
        callback = RaycastCallback()
        self.world.RayCast(callback, start_point, end_point)
        return callback

    def calculate_state(self, action):
        angle_25_degrees = math.pi / 6
        angle_50_degrees = math.pi / 3.5
        angle_75_degrees = math.pi / 2.4

        angle = self.ship_body.angle
        self.raycasts = [
            calculate_raycast(self.ship_body, angle, 200),
            calculate_raycast(self.ship_body, angle + angle_50_degrees, 250),
            calculate_raycast(self.ship_body, angle - angle_50_degrees, 250),
            calculate_raycast(self.ship_body, angle + angle_25_degrees, 250),
            calculate_raycast(self.ship_body, angle - angle_25_degrees, 250),
            calculate_raycast(self.ship_body, angle + angle_75_degrees, 250),
            calculate_raycast(self.ship_body, angle - angle_75_degrees, 250),
        ]

        callbacks = []
        for pos1, pos2 in self.raycasts:
            callbacks.append(self.perform_raycast(pos1, pos2))

        for callback in callbacks:
            if callback.hit:
                self.raycast_hits.append(callback.point)
            else:
                self.raycast_hits.append(None)

        # calc distance for each raycast hit
        distances = []
        for hit in self.raycast_hits:
            if hit is not None:
                distance = math.sqrt(
                    (self.ship_body.position[0] - hit[0]) ** 2
                    + (self.ship_body.position[1] - hit[1]) ** 2
                )
                distances.append(distance - 20)
            else:
                distances.append(9999)

        self.distance_to_landing_zone = abs(
            self.ship_body.position[0] - LANDING_ZONE_SIZE / 2
        ) + abs(self.ship_body.position[1] - LANDING_ZONE_SIZE / 2)

        reward = 0

        reward -= 0.1 * abs(action[0])
        reward -= 0.05 * abs(action[1])

        if self.distance_to_landing_zone < LANDING_ZONE_SIZE / 2:
            self.terminate = True
            reward += 250

        # getting closer to the landing zone gives reward
        for distance_reward in self.distance_step_rewards:
            if self.distance_to_landing_zone < distance_reward:
                reward += 5
                self.distance_step_rewards.remove(distance_reward)

        if (
            self.ship_body.position[0] < 0
            or self.ship_body.position[0] > SCREEN_WIDTH
            or self.ship_body.position[1] < 0
            or self.ship_body.position[1] > SCREEN_HEIGHT
        ):
            self.terminate = True
            reward -= 250

        if self.step_count > 2000:
            self.terminate = True
            reward -= 50

        self.state = [
            self.ship_body.position[0],
            self.ship_body.position[1],
            self.ship_body.linearVelocity[0],
            self.ship_body.linearVelocity[1],
            self.ship_body.angle,
            self.distance_to_landing_zone,
            *distances,
        ]

        return reward

    def step(self, action):
        if not self.continuous:
            if action == 0:
                action = [0, 0]
            elif action == 1:
                action = [0, 1]
            elif action == 2:
                action = [1, 0]
            elif action == 3:
                action = [-1, 0]

        self.step_count += 1
        self.raycasts = []
        self.raycast_hits = []

        # foward thrust
        angle = self.ship_body.angle
        rotated_angle = angle - math.pi / 2

        force = (
            vec2(-math.cos(rotated_angle), math.sin(rotated_angle))
            * SHIP_THRUST
            * (action[1])
        )
        self.ship_body.ApplyForceToCenter(force, True)
        self.thruster_active = action[1] > 0

        # rotation
        self.ship_body.angularVelocity += ROTATE_SPEED * action[0]
        self.ship_body.angularVelocity = self.ship_body.angularVelocity * 0.99

        self.world.Step(TIME_STEP, 6, 2)

        reward = self.calculate_state(action)
        self.total_reward += reward

        if self.render_mode == "human":
            self.render()

        return (
            np.array(self.state, dtype=np.float32),
            reward,
            self.terminate,
            False,
            {},
        )

    def draw_spaceship(self):
        # Get the body's rotation angle in radians
        angle = self.ship_body.angle

        # Get the ship's position in pixels
        ship_pos = (
            self.ship_body.position[0],
            SCREEN_HEIGHT - self.ship_body.position[1],
        )

        # Rotate each vertex around the origin and then translate to the spaceship's position
        vertices = [
            rotate_point((v[0], v[1]), angle)
            for v in self.ship_body.fixtures[0].shape.vertices
        ]
        vertices = [(v[0] + ship_pos[0], v[1] + ship_pos[1]) for v in vertices]

        pygame.draw.polygon(self.screen, YELLOW, vertices)

        if self.thruster_active:
            # Thruster flame is behind the spaceship, so calculate the opposite direction
            flame_offset_distance = (
                15  # Adjust this to control how far behind the flame is
            )

            # Compute the flame position by offsetting the ship position in the opposite direction
            flame_pos_x = ship_pos[0] + math.sin(angle) * flame_offset_distance
            flame_pos_y = ship_pos[1] - math.cos(angle) * flame_offset_distance

            # Shrink the flame relative to the ship and rotate opposite to the ship's direction (angle + math.pi)
            vertices = [
                rotate_point((v[0], v[1]), angle + math.pi)
                for v in [(5, -5), (-5, -5), (0, 10)]
            ]
            vertices = [(v[0] + flame_pos_x, v[1] + flame_pos_y) for v in vertices]

            pygame.draw.polygon(self.screen, RED, vertices)

    def draw_asteroids(self):
        for body, radius in self.asteroids:
            pygame.draw.circle(
                self.screen,
                GRAY,
                (body.position[0], SCREEN_HEIGHT - body.position[1]),
                radius,
            )

    def draw_landing_zone(self):
        # Define the rectangle in Pygame's coordinate system (top-left corner)
        rect = pygame.Rect(
            0,  # Center the rectangle on the body's position
            self.screen.get_height() - LANDING_ZONE_SIZE,  # Bottom of the screen
            LANDING_ZONE_SIZE,
            LANDING_ZONE_SIZE,
        )

        pygame.draw.rect(self.screen, GREEN, rect, 3)

    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Spaceship in Asteroid Field")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.terminate = False
                self._destroy()
                exit()

        self.surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.screen.fill(BLACK)

        for pos1, pos2 in self.raycasts:
            pygame.draw.line(
                self.screen,
                (255, 0, 0),
                (pos1[0], SCREEN_HEIGHT - pos1[1]),
                (pos2[0], SCREEN_HEIGHT - pos2[1]),
            )

        for point in self.raycast_hits:
            if point is not None:
                hit_point = (
                    point[0],
                    self.screen.get_height() - point[1],
                )
                pygame.draw.circle(self.screen, (250, 0, 0), hit_point, 5)

        self.draw_landing_zone()
        self.draw_asteroids()
        self.draw_spaceship()

        font = pygame.font.Font(None, 32)
        debug_pos = f"Pos: {self.ship_body.position[0]:.2f}, {SCREEN_HEIGHT - self.ship_body.position[1]:.2f}"
        debug_speed = f"Speed: {self.ship_body.linearVelocity[0]:.2f}, {self.ship_body.linearVelocity[1]:.2f}"
        debug_distance = f"Distance: {self.distance_to_landing_zone:.2f}"
        debug_steps = f"Step: {self.step_count}"
        debug_reward = f"Reward: {self.total_reward:.2f}"
        text_pos = font.render(debug_pos, True, WHITE)
        text_speed = font.render(debug_speed, True, WHITE)
        text_distance = font.render(debug_distance, True, WHITE)
        text_step = font.render(debug_steps, True, WHITE)
        text_reward = font.render(debug_reward, True, WHITE)
        self.screen.blit(text_pos, (0, 0))
        self.screen.blit(text_speed, (0, 20))
        self.screen.blit(text_distance, (0, 40))
        self.screen.blit(text_step, (0, 60))
        self.screen.blit(text_reward, (0, 80))

        self.clock.tick(TARGET_FPS)

        pygame.display.flip()


def main():
    asteroids = AsteroidDodger(render_mode="human")

    asteroids.render()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        action = np.array([0, 0])
        if keys[pygame.K_UP]:
            action[1] = 1
        elif keys[pygame.K_DOWN]:
            action[1] = -1
        if keys[pygame.K_LEFT]:
            action[0] = -1
        elif keys[pygame.K_RIGHT]:
            action[0] = 1

        obs, reward, terminate, k, info = asteroids.step(action)

        if terminate:
            asteroids.reset()


if __name__ == "__main__":
    main()
