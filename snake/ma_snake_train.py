import argparse
import os

import gymnasium as gym
from colorama import Fore
from stable_baselines3 import DQN, PPO, SAC, TD3, A2C
from gymnasium.envs.registration import register
from gymnasium.wrappers import RecordVideo

register(
    id="ma_snake_env",
    entry_point="ma_snake_env:SnakeGameEnv",
)

def train_model(timesteps=250000, iters=1, replace=False, algo=DQN, algo_name="DQN"):
    print("Training model for ", timesteps, " timesteps over ", iters, " iterations.")
    model_dir = "models/"
    model_name = algo_name + "_ma_snake"
    os.makedirs(model_dir, exist_ok=True)
    env = gym.make("ma_snake_env")

    model_path = f"{model_dir}{model_name}{timesteps*iters}"

    if replace and os.path.exists(model_path + ".zip"):
        print("Replacing existing model.")
        os.remove(model_path + ".zip")

    model = algo(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="tlogs/",
        # exploration_fraction=0.1,
        # exploration_final_eps=0.05,
    )
    i = 0
    while i < iters:
        i += 1
        model.learn(
            total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=True
        )
        model.save(f"{model_dir}{model_name}{timesteps*i}")


def test_model(model_name: str, record: bool, algo=DQN):
    print(Fore.CYAN + "Testing model:", Fore.YELLOW + model_name)
    render_mode = "rgb_array" if record else "human"
    env = gym.make("ma_snake_env", render_mode=render_mode)

    if record:
        env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)

    model = algo.load(model_name, env=env)

    obs = env.reset()[0]
    done = False
    score = 0
    if record: 
        env.start_video_recorder();
    episodes = 0
    while episodes < 10:
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        score += reward
        if done:
            print(Fore.GREEN + "DONE")
            print(Fore.MAGENTA + "Score:", Fore.YELLOW + str(score))
            score = 0
            episodes += 1
            env.reset()
            
    if record:
        env.close_video_recorder();
    env.close();


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test DQN model for Snake game."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=250000,
        help="Number of timesteps for training.",
    )
    parser.add_argument(
        "--iters", type=int, default=1, help="Number of iterations for training."
    )
    parser.add_argument(
        "-r", "--replace", action="store_true", help="Replace and overwrite the existing model."
    )
    parser.add_argument(
        "--record", action="store_true", help="Record the video output of the game."
    )
    parser.add_argument(
        "--rl-algo", default="DQN", help="What RL algorithm to use."
    )
    args = parser.parse_args()

    if args.rl_algo == "DQN":
        rl_algo = DQN
    elif args.rl_algo == "PPO":
        rl_algo = PPO
    elif args.rl_algo == "SAC":
        rl_algo = SAC
    elif args.rl_algo == "TD3":
        rl_algo = TD3
    elif args.rl_algo == "A2C":
        rl_algo = A2C
    else:
        print(Fore.RED + args.rl_algo, "is not a valid RL algorithm. Pick one of DQN, PPO, SAC, TD3, A2C.")
        exit(0)

    train_model(timesteps=args.timesteps, iters=args.iters, replace=args.replace, algo=rl_algo, algo_name=args.rl_algo)
    test_model(f"models/{args.rl_algo}_ma_snake{args.timesteps * args.iters}", record=args.record, algo=rl_algo)