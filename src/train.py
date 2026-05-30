import os
import argparse
import gymnasium as gym
from typing import Dict, Any, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from src.environment import USTDriverEnv

def make_env(scenario: str, behaviour_profile: str, seed: int = 0) -> Callable[[], gym.Env]:
    """
    Utility function for multiprocessed env.
    """
    def _init() -> gym.Env:
        env = USTDriverEnv({
            "scenario": scenario,
            "behaviour_profile": behaviour_profile,
            "use_render": False
        })
        # Seed gymnasium space if supported
        env.action_space.seed(seed)
        return env
    return _init

def train(
    scenario: str,
    profile: str,
    total_timesteps: int,
    num_envs: int,
    tb_log_dir: str,
    model_save_path: str
) -> None:
    """
    Main training function using PPO and Stable-Baselines3.
    """
    print(f"Initializing training for Scenario: {scenario}, Profile: {profile}")
    print(f"Parallel Environments: {num_envs}, Total Timesteps: {total_timesteps}")
    
    # 1. Create parallel environment wrapper
    env_fns = [make_env(scenario, profile, i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    
    # 2. Define the exact policy network architecture:
    # 2-layer MLP (256 units), separate policy (pi) and value (vf) heads.
    policy_kwargs: Dict[str, Any] = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    # 3. Instantiate the PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_log_dir
    )
    
    # 4. Set up checkpoint callback to save model during training
    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 5, 10000),
        save_path=os.path.dirname(model_save_path),
        name_prefix=f"{profile}_{scenario}_ppo_checkpoint"
    )
    
    # 5. Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # 6. Save the final policy weights
    model.save(model_save_path)
    print(f"Successfully trained and saved model to {model_save_path}")
    
    # Close vector environment
    vec_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for UST Automotive Data Science assignment.")
    parser.add_argument("--scenario", type=str, default="S1", choices=["S1", "S2", "S3", "S4", "S5"],
                        help="Scenario to train on (S1-S5)")
    parser.add_argument("--profile", type=str, default="normal", choices=["normal", "aggressive", "cautious"],
                        help="Behaviour profile (normal, aggressive, cautious)")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total timesteps to run training (default: 50,000 for quick run. Use 2,000,000 for Part 1 scale)")
    parser.add_argument("--envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    tb_log = f"logs/tb_{args.profile}_{args.scenario}"
    model_path = f"models/{args.profile}_{args.scenario}_ppo"
    
    train(
        scenario=args.scenario,
        profile=args.profile,
        total_timesteps=args.timesteps,
        num_envs=args.envs,
        tb_log_dir=tb_log,
        model_save_path=model_path
    )
