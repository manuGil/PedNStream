# -*- coding: utf-8 -*-
"""
Train PPO agents using Stable-Baselines3 library.

This script uses the industry-standard SB3 library to train agents on the 
PedNetParallelEnv environment, similar to train_rl.py but using SB3's PPO.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
from rl import PedNetParallelEnv
# Note: Using SB3's VecNormalize instead of RunningNormalizeWrapper
from datetime import datetime
import os
from gymnasium import spaces as gym_spaces
import torch

# Import SB3
try:
    from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
    from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, VecNormalize, VecFrameStack
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
except ImportError:
    print("ERROR: Missing dependencies. Please install:")
    print("  pip install stable-baselines3")
    sys.exit(1)

# Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class WandbCallback(BaseCallback):
    """
    Custom callback for logging metrics to wandb, similar to PPO_backup training.
    Tracks episode returns (normalized and true) and logs per-agent metrics.
    """
    def __init__(self, verbose=0, use_wandb=True, env_wrapper=None):
        super().__init__(verbose)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.env_wrapper = env_wrapper  # Reference to PedNetSB3Wrapper to extract true rewards
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_true_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Called at each step. Check for episode completion."""
        if not self.use_wandb or not wandb.run:
            return True
        
        # Check if episode is done by looking at infos
        infos = self.locals.get('infos', [])
        if infos and len(infos) > 0:
            info = infos[0]
            # VecMonitor adds 'episode' key when episode is done
            if 'episode' in info:
                episode_info = info['episode']
                episode_reward = episode_info.get('r', 0.0)
                episode_length = episode_info.get('l', 0)
                
                self.episode_count += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Extract true reward from combined_info (stored by PedNetSB3Wrapper)
                # The wrapper stores total_true_reward in combined_info
                true_reward = episode_reward  # Default to normalized reward
                if 'total_true_reward' in info:
                    true_reward = info['total_true_reward']
                elif 'individual_true_rewards' in info:
                    # Fallback: sum individual true rewards if total not available
                    true_reward = sum(info['individual_true_rewards'].values())
                
                self.episode_true_rewards.append(true_reward)
                
                # Log to wandb (similar to PPO_backup)
                log_dict = {
                    'episode': self.episode_count,
                    'avg_normalized_return': episode_reward,
                    'avg_true_return': true_reward,
                    'total_normalized_return': episode_reward,  # Single agent = total
                    'total_true_return': true_reward,
                    'episode_steps': episode_length
                }
                
                # Log rolling averages
                if len(self.episode_rewards) > 0:
                    log_dict['avg_normalized_return_100'] = np.mean(self.episode_rewards[-100:])
                    if len(self.episode_true_rewards) > 0:
                        log_dict['avg_true_return_100'] = np.mean(self.episode_true_rewards[-100:])
                
                wandb.log(log_dict)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        pass


import gymnasium as gym


class PedNetSB3Wrapper(gym.Env):
    """
    Gymnasium-compatible wrapper to convert PedNetParallelEnv to single-agent format for SB3.
    
    Treats all agents as one by concatenating observations and actions.
    Supports delta actions by converting to absolute actions.
    """
    
    metadata = {"render_modes": ["human", "animate"]}
    
    def __init__(self, env, delta_actions=False, reset_options=None):
        super().__init__()
        
        self.env = env
        self.agents = env.possible_agents
        self.delta_actions = delta_actions
        # Default options for reset if none provided
        self.reset_options = reset_options if reset_options is not None else {'randomize': True}
        
        # Build concatenated observation space
        obs_spaces = [env.observation_space(agent) for agent in self.agents]
        total_obs_dim = sum(space.shape[0] for space in obs_spaces)
        self.observation_space = gym_spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        # Build concatenated action space
        act_spaces = [env.action_space(agent) for agent in self.agents]
        total_act_dim = sum(space.shape[0] for space in act_spaces)
        act_lows = np.concatenate([space.low for space in act_spaces])
        act_highs = np.concatenate([space.high for space in act_spaces])
        self.action_space = gym_spaces.Box(
            low=act_lows,
            high=act_highs,
            shape=(total_act_dim,),
            dtype=np.float32
        )
        
        # Store dimensions for splitting actions
        self.act_dims = [space.shape[0] for space in act_spaces]
        self.obs_dims = [space.shape[0] for space in obs_spaces]
        
        # Store action bounds for delta action conversion
        self.act_lows = act_lows
        self.act_highs = act_highs
        
        # Track previous observations for delta actions
        self.prev_obs = None
    
    def reset(self, seed=None, options=None):
        """Reset environment and return concatenated observation.
        Note: seed parameter is ignored here - seed should be set at env construction.
        VecEnv calls reset without options, so we use self.reset_options as default.
        """
        # Use default options if none provided (VecEnv calls reset without options)
        if options is None:
            options = self.reset_options
            
        obs_dict, infos = self.env.reset(options=options)
        
        # Concatenate observations
        obs_list = [obs_dict[agent] for agent in self.agents]
        concat_obs = np.concatenate(obs_list, dtype=np.float32)
        
        # Store for delta action conversion
        self.prev_obs = concat_obs.copy()
        
        # Combine infos (just use first agent's info for simplicity)
        combined_info = infos[self.agents[0]] if self.agents else {}
        
        return concat_obs, combined_info
    
    def step(self, action):
        """Step environment with split actions. Supports delta actions."""
        # Split concatenated action into individual agent actions
        actions_dict = {}
        obs_start_idx = 0
        action_start_idx = 0
        
        for i, agent in enumerate(self.agents):
            # Extract this agent's observation slice
            obs_end_idx = obs_start_idx + self.obs_dims[i]
            agent_obs = self.prev_obs[obs_start_idx:obs_end_idx] if self.prev_obs is not None else None
            
            # Extract this agent's action slice
            action_end_idx = action_start_idx + self.act_dims[i]
            agent_delta_action = action[action_start_idx:action_end_idx]
            
            # Convert delta to absolute action if needed
            if self.delta_actions and agent_obs is not None:
                # Similar to train_rl.py: reshape obs to (act_dim, features_per_link)
                # and extract gate widths from last column
                try:
                    act_dim = self.act_dims[i]
                    features_per_link = len(agent_obs) // act_dim
                    if features_per_link > 0:
                        obs_reshaped = agent_obs.reshape(act_dim, features_per_link)
                        current_gate_widths = obs_reshaped[:, -1]  # Last feature per link
                        absolute_action = current_gate_widths + agent_delta_action
                    else:
                        # Fallback: assume gate widths are at the end
                        absolute_action = agent_obs[-act_dim:] + agent_delta_action
                    
                    # Clip to action bounds
                    agent_act_low = self.act_lows[action_start_idx:action_end_idx]
                    agent_act_high = self.act_highs[action_start_idx:action_end_idx]
                    actions_dict[agent] = np.clip(absolute_action, agent_act_low, agent_act_high)
                except Exception:
                    # Fallback: treat as absolute action
                    agent_act_low = self.act_lows[action_start_idx:action_end_idx]
                    agent_act_high = self.act_highs[action_start_idx:action_end_idx]
                    actions_dict[agent] = np.clip(agent_delta_action, agent_act_low, agent_act_high)
            else:
                # Use absolute actions directly
                agent_act_low = self.act_lows[action_start_idx:action_end_idx]
                agent_act_high = self.act_highs[action_start_idx:action_end_idx]
                actions_dict[agent] = np.clip(agent_delta_action, agent_act_low, agent_act_high)
            
            obs_start_idx = obs_end_idx
            action_start_idx = action_end_idx
        
        # Step environment
        obs_dict, rewards, terms, truncs, infos = self.env.step(actions_dict)
        
        # Concatenate observations
        obs_list = [obs_dict[agent] for agent in self.agents]
        concat_obs = np.concatenate(obs_list, dtype=np.float32)
        
        # Update previous observation for next delta action conversion
        self.prev_obs = concat_obs.copy()
        
        # Sum rewards (cooperative task)
        total_reward = sum(rewards.values())
        
        # Any agent done = episode done
        done = any(terms.values()) or any(truncs.values())
        terminated = any(terms.values())
        truncated = any(truncs.values())
        
        # Combine infos
        combined_info = infos[self.agents[0]].copy() if self.agents and self.agents[0] in infos else {}
        combined_info['individual_rewards'] = rewards
        
        # Store true rewards (rewards from base env are already true/unnormalized)
        # VecNormalize will normalize rewards after this wrapper, so we save true rewards here
        combined_info['individual_true_rewards'] = dict(rewards)
        combined_info['total_true_reward'] = total_reward  # Sum of true rewards
        
        return concat_obs, total_reward, terminated, truncated, combined_info
    
    def render(self):
        """Render environment."""
        return self.env.render()
    
    def close(self):
        """Close environment."""
        return self.env.close()


def make_env(dataset="long_corridor", obs_mode="option3",
             randomize=False, seed=None, action_gap=1, delta_actions=False):
    """
    Create and wrap a PedNet environment for SB3.
    
    Note: Normalization is handled by SB3's VecNormalize wrapper, not here.
    
    Args:
        dataset: Network dataset name
        obs_mode: Observation mode ("option1", "option2", "option3", "option4")
        randomize: Whether to randomize network at each reset
        seed: Random seed (set at env construction)
        action_gap: Number of steps between applying actions
        delta_actions: Whether to use delta actions
    
    Returns:
        Wrapped environment compatible with SB3
    """
    def _init():
        # Create base environment
        base_env = PedNetParallelEnv(
            dataset=dataset,
            normalize_obs=False,  # Normalization handled by VecNormalize
            obs_mode=obs_mode,
            render_mode=None,  # No rendering during training
            verbose=False,
            action_gap=action_gap,
            seed=seed
        )
        
        # Wrap with custom wrapper for SB3 (no RunningNormalizeWrapper - use VecNormalize instead)
        # Pass reset_options to control randomization since VecEnv.reset() doesn't take args
        env = PedNetSB3Wrapper(base_env, delta_actions=delta_actions, reset_options={'randomize': randomize})
        
        return env
    
    return _init


def train_sb3_agent(
    algo="ppo",
    dataset=None,
    total_timesteps=100_000,
    learning_rate=3e-4,
    obs_mode="option3",
    action_gap=1,
    delta_actions=True,
    norm_obs=True,
    norm_reward=False,
    randomize=True,
    seed=100,
    save_dir="rl_models_sb3",
    eval_freq=10_000,
    num_val_episodes=5,
    use_wandb=True,
    n_stack=1,
    # Algorithm-specific hyperparameters
    **algo_kwargs
):
    """
    Train agents using Stable-Baselines3 (supports PPO, SAC, TD3, A2C, DDPG).
    
    Args:
        algo: Algorithm name ("ppo", "sac", "td3", "a2c", "ddpg")
        dataset: Network dataset to use
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        obs_mode: Observation mode ("option1", "option2", "option3", "option4")
        action_gap: Number of steps between applying actions
        delta_actions: Whether to use delta actions
        norm_obs: Whether to normalize observations
        norm_reward: Whether to normalize rewards
        randomize: Whether to randomize environment at reset
        seed: Random seed
        save_dir: Directory to save models
        eval_freq: Evaluation frequency (timesteps) - SB3's EvalCallback will run validation
        num_val_episodes: Number of episodes for validation (used by EvalCallback)
        use_wandb: Whether to log to wandb
        n_stack: Number of frames to stack (1 = no stacking)
        **algo_kwargs: Algorithm-specific hyperparameters (e.g., n_steps, batch_size for PPO)
    
    Returns:
        model: Trained model
        run_name: Model directory name (e.g., "sb3_ppo_agents_butterfly_scC")
    """
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Validate algorithm
    algo = algo.lower()
    supported_algos = ["ppo", "sac", "td3", "a2c", "ddpg"]
    if algo not in supported_algos:
        raise ValueError(f"Unsupported algorithm: {algo}. Supported: {supported_algos}")
    
    print("=" * 70)
    print(f"Training {algo.upper()} Agents using Stable-Baselines3")
    print("=" * 70)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Use fixed name like train_rl.py: {algo}_agents_{dataset}
    run_name = f"sb3_{algo}_agents_{dataset}"
    
    print(f"\nRun name: {run_name}")
    print(f"Algorithm: {algo.upper()}")
    print(f"Dataset: {dataset}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Observation mode: {obs_mode}")
    print(f"Action gap: {action_gap}")
    print(f"Delta actions: {delta_actions}")
    print(f"Normalize observations: {norm_obs}")
    print(f"Normalize rewards: {norm_reward}")
    print(f"Randomize: {randomize}")
    print(f"Seed: {seed}")
    
    # Initialize wandb if available
    if use_wandb and WANDB_AVAILABLE:
        if wandb.run is None:
            wandb.init(
                project="crowd-control-rl",
                name=f"sb3-{algo}-{dataset}",
                config={
                    "algorithm": algo,
                    "dataset": dataset,
                    "total_timesteps": total_timesteps,
                    "learning_rate": learning_rate,
                    "obs_mode": obs_mode,
                    "action_gap": action_gap,
                    "delta_actions": delta_actions,
                    "norm_obs": norm_obs,
                    "norm_reward": norm_reward,
                    "randomize": randomize,
                    "seed": seed,
                    **algo_kwargs
                }
            )
    
    # Create training environment
    print("\nCreating training environment...")
    env_fn = make_env(
        dataset=dataset,
        obs_mode=obs_mode,
        randomize=randomize,
        seed=seed,
        action_gap=action_gap,
        delta_actions=delta_actions
    )
    
    # Wrap in DummyVecEnv (SB3 requirement)
    env = DummyVecEnv([env_fn])
    
    # Get reference to wrapper for callback
    env_wrapper = env.envs[0]  # PedNetSB3Wrapper
    
    # Wrap with VecNormalize for observation and reward normalization (SB3's built-in)
    # Note: VecNormalize should be BEFORE VecFrameStack (normalize first, then stack)
    env = VecNormalize(
        env,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        training=True
    )
    
    # Wrap with VecFrameStack (Stacking happens AFTER normalization, BEFORE monitor)
    # Order: DummyVecEnv -> VecNormalize -> VecFrameStack -> VecMonitor
    if n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack)
        print(f"Stacked {n_stack} frames. New observation shape: {env.observation_space.shape}")
    
    # Wrap with monitor for logging (outermost wrapper to track episode stats)
    env = VecMonitor(env)
    
    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create evaluation environment for SB3's standard validation
    print("\nCreating evaluation environment...")
    eval_env_fn = make_env(
        dataset=dataset,
        obs_mode=obs_mode,
        randomize=False,  # Deterministic for evaluation
        seed=seed,
        action_gap=action_gap,
        delta_actions=delta_actions
    )
    eval_env = DummyVecEnv([eval_env_fn])
    
    # Use VecNormalize for eval env too
    # norm_reward=False for evaluation to get true rewards
    eval_env = VecNormalize(
        eval_env,
        norm_obs=norm_obs,
        norm_reward=False,  # Don't normalize rewards for evaluation
        clip_obs=10.0,
        training=False  # Don't update stats during evaluation
    )
    
    # Apply same frame stacking to eval env (same order as training)
    if n_stack > 1:
        eval_env = VecFrameStack(eval_env, n_stack=n_stack)
    
    # Wrap with monitor for logging (outermost)
    eval_env = VecMonitor(eval_env)
    
    # Create callbacks
    # Use SB3's standard EvalCallback for validation - saves only the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, run_name, "best_model"),
        log_path=os.path.join(save_dir, run_name, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=num_val_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Add wandb callback
    wandb_callback = WandbCallback(verbose=1, use_wandb=use_wandb, env_wrapper=env_wrapper)
    
    callbacks = [eval_callback, wandb_callback]
    
    # Create model based on algorithm
    print(f"\nCreating {algo.upper()} model...")
    
    # Default hyperparameters for each algorithm
    default_kwargs = {
        "ppo": {
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "sac": {
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "ent_coef": "auto",
        },
        "td3": {
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
        },
        "a2c": {
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "ddpg": {
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
        },
    }
    
    # Merge default kwargs with user-provided kwargs
    algo_params = default_kwargs.get(algo, {}).copy()
    algo_params.update(algo_kwargs)
    algo_params["learning_rate"] = learning_rate
    algo_params["verbose"] = 1
    algo_params["device"] = "auto"
    
    # Create model
    if algo == "ppo":
        model = PPO(policy="MlpPolicy", env=env, **algo_params)
    elif algo == "sac":
        model = SAC(policy="MlpPolicy", env=env, **algo_params)
    elif algo == "td3":
        model = TD3(policy="MlpPolicy", env=env, **algo_params)
    elif algo == "a2c":
        model = A2C(policy="MlpPolicy", env=env, **algo_params)
    elif algo == "ddpg":
        model = DDPG(policy="MlpPolicy", env=env, **algo_params)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    print(f"Model created with policy architecture:")
    print(f"  Algorithm: {algo.upper()}")
    print(f"  Policy: MlpPolicy (default: [64, 64] hidden layers)")
    print(f"  Device: {model.device}")
    if algo_params:
        print(f"  Hyperparameters: {', '.join(f'{k}={v}' for k, v in algo_params.items() if k not in ['verbose', 'device'])}")
    
    # Train model (SB3 handles validation automatically via EvalCallback)
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print("\nSB3 will automatically run validation every {} timesteps".format(eval_freq))
    print("Best model will be saved to: {}/best_model/".format(os.path.join(save_dir, run_name)))
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save VecNormalize stats (SB3's built-in normalization)
    # Handle wrapper chain: VecMonitor -> VecFrameStack (optional) -> VecNormalize -> DummyVecEnv
    try:
        # Get VecNormalize from the wrapper chain
        # Helper function to find VecNormalize in wrapper chain
        def find_vec_normalize(venv):
            if isinstance(venv, VecNormalize):
                return venv
            elif hasattr(venv, 'venv'):
                return find_vec_normalize(venv.venv)
            return None
        
        vec_normalize = find_vec_normalize(env)
        if vec_normalize is not None:
            norm_stats_path = os.path.join(save_dir, run_name, "vec_normalize.pkl")
            vec_normalize.save(norm_stats_path)
            print(f"VecNormalize stats saved to: {norm_stats_path}")
        else:
            print("Warning: VecNormalize not found in wrapper chain")
    except Exception as e:
        print(f"Warning: Could not save VecNormalize stats: {e}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    # Close wandb if it was initialized
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {save_dir}/{run_name}")
    best_model_path = os.path.join(save_dir, run_name, "best_model", "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"✅ Best model saved to: {best_model_path}")
    else:
        print(f"⚠️  Warning: Best model not found at {best_model_path}")
        print("   This may happen if no validation was performed or validation failed.")
    print("\nTo load and evaluate the model:")
    print(f'  from stable_baselines3 import {algo.upper()}')
    print(f'  model = {algo.upper()}.load("{best_model_path}")')
    print('  obs = env.reset()')
    print('  action, _states = model.predict(obs, deterministic=True)')
    
    return model, run_name


# Backward compatibility alias
def train_sb3_ppo(*args, **kwargs):
    """Backward compatibility alias for train_sb3_agent with algo='ppo'."""
    return train_sb3_agent(algo="ppo", *args, **kwargs)


def evaluate_model(model_path, dataset="long_corridor", obs_mode="option3", 
                   n_episodes=10, action_gap=1, delta_actions=True,
                   norm_obs=True, seed=42, randomize=False, save_dir=None, algo=None,
                   vec_normalize_path=None, n_stack=1):
    """
    Evaluate a trained SB3 model using the full VecEnv stack (VecNormalize + VecFrameStack).
    Saves results in format compatible with evaluate_and_visualize.py.
    
    Args:
        model_path: Path to saved model
        dataset: Dataset to evaluate on
        obs_mode: Observation mode
        n_episodes: Number of episodes to run
        action_gap: Number of steps between applying actions
        delta_actions: Whether model uses delta actions
        norm_obs: Whether to use normalized observations (via VecNormalize)
        seed: Random seed
        randomize: Whether to randomize environment
        save_dir: Directory to save evaluation results (compatible with evaluate_and_visualize.py)
        algo: Algorithm name (auto-detected from model_path if None)
        vec_normalize_path: Path to saved VecNormalize stats (auto-detected if None)
        n_stack: Number of frames to stack (must match training)
    """
    from rl import PedNetParallelEnv
    
    print("=" * 70)
    print("Evaluating SB3 Model")
    print("=" * 70)
    
    # Auto-detect algorithm from path if not provided
    if algo is None:
        if "ppo" in model_path.lower():
            algo = "ppo"
        elif "sac" in model_path.lower():
            algo = "sac"
        elif "td3" in model_path.lower():
            algo = "td3"
        elif "a2c" in model_path.lower():
            algo = "a2c"
        elif "ddpg" in model_path.lower():
            algo = "ddpg"
        else:
            algo = "ppo"  # Default to PPO
    
    # Load model
    print(f"\nLoading {algo.upper()} model from: {model_path}")
    algo_class = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C, "ddpg": DDPG}.get(algo.lower(), PPO)
    model = algo_class.load(model_path)
    
    # Auto-detect VecNormalize path if not provided
    if vec_normalize_path is None and norm_obs:
        # Try to find vec_normalize.pkl in the same directory as the model
        model_dir = os.path.dirname(os.path.dirname(model_path))  # Go up from best_model/
        potential_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(potential_path):
            vec_normalize_path = potential_path
            print(f"Found VecNormalize stats at: {vec_normalize_path}")
    
    # Create evaluation environment wrapped in DummyVecEnv
    # We pass reset_options to control randomization since VecEnv.reset() doesn't take args
    def make_eval_env():
        base_env = PedNetParallelEnv(
            dataset=dataset,
            normalize_obs=False,
            obs_mode=obs_mode,
            render_mode=None,
            verbose=False,
            action_gap=action_gap,
            seed=seed
        )
        return PedNetSB3Wrapper(base_env, delta_actions=delta_actions, reset_options={'randomize': randomize})

    env = DummyVecEnv([make_eval_env])
    
    # Load VecNormalize stats if available
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats from: {vec_normalize_path}")
    elif norm_obs:
        print("Warning: norm_obs=True but no VecNormalize stats found. Evaluation might be incorrect.")

    # Wrap with VecFrameStack
    if n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack)
        print(f"Stacked {n_stack} frames for evaluation.")
    
    # Get agent info from the inner env
    # Chain: VecFrameStack (optional) -> VecNormalize (optional) -> DummyVecEnv -> PedNetSB3Wrapper
    # Helper function to get the PedNetSB3Wrapper
    def get_inner_wrapper(env):
        """Navigate through wrapper chain to get PedNetSB3Wrapper."""
        if isinstance(env, VecFrameStack):
            return get_inner_wrapper(env.venv)
        elif isinstance(env, VecNormalize):
            return get_inner_wrapper(env.venv)
        elif isinstance(env, DummyVecEnv):
            return env.envs[0]
        else:
            return env
    
    inner_wrapper = get_inner_wrapper(env)
    agents_list = inner_wrapper.agents
    
    episode_rewards = []
    episode_lengths = []
    all_episode_true_returns = []
    
    for episode in range(n_episodes):
        # Reset environment
        obs = env.reset()
        
        episode_reward = 0
        episode_length = 0
        episode_true_returns = {agent_id: 0.0 for agent_id in agents_list}
        done = False
        
        while not done:
            # Predict action (obs is already stacked/normalized by VecEnv wrappers)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, rewards, dones, infos = env.step(action)
            
            # Extract info (VecEnv returns a list of infos, we have 1 env)
            info = infos[0]
            reward = rewards[0]
            done = dones[0]
            
            episode_reward += reward
            episode_length += 1
            
            # Track true rewards (stored in info by PedNetSB3Wrapper)
            if 'individual_true_rewards' in info:
                for agent_id, true_r in info['individual_true_rewards'].items():
                    episode_true_returns[agent_id] += true_r
            else:
                # Fallback: if true rewards not in info, we can't track them separately
                # This shouldn't happen if PedNetSB3Wrapper is working correctly
                pass
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_episode_true_returns.append(episode_true_returns)
        
        avg_true_return = np.mean(list(episode_true_returns.values()))
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"TrueAvgReturn={avg_true_return:.2f}, "
              f"Length={episode_length}")
        
        # Save simulation if save_dir is provided (compatible with evaluate_and_visualize.py)
        if save_dir is not None:
            # PedNetParallelEnv.save() uses base_dir="../outputs" and prepends it to simulation_dir
            # So we need to extract the relative path from "outputs/" onwards
            from pathlib import Path
            save_path = Path(save_dir)
            
            # If save_dir contains "outputs", extract everything after "outputs/"
            if "outputs" in save_path.parts:
                try:
                    outputs_idx = save_path.parts.index("outputs")
                    relative_path = str(Path(*save_path.parts[outputs_idx + 1:]))
                except (ValueError, IndexError):
                    relative_path = str(save_path)
            else:
                relative_path = str(save_path)
            
            run_save_dir = f"{relative_path}_run{episode + 1}" if n_episodes > 1 else relative_path
            
            # Access the base PedNetParallelEnv to save through the wrapper chain
            base_env = inner_wrapper.env  # PedNetSB3Wrapper -> PedNetParallelEnv
            base_env.save(run_save_dir)
            if episode == 0 or (episode + 1) % 5 == 0:
                print(f"Saved run {episode + 1} to outputs/{run_save_dir}")
    
    # env.close()
    
    # Calculate statistics
    avg_reward_mean = np.mean(episode_rewards)
    avg_reward_std = np.std(episode_rewards)
    avg_true_returns = [np.mean(list(ep_returns.values())) for ep_returns in all_episode_true_returns]
    avg_true_return_mean = np.mean(avg_true_returns)
    avg_true_return_std = np.std(avg_true_returns)
    
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Mean reward: {avg_reward_mean:.2f} ± {avg_reward_std:.2f}")
    print(f"Mean true return: {avg_true_return_mean:.2f} ± {avg_true_return_std:.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_true_returns': all_episode_true_returns,
        'avg_reward': avg_reward_mean,
        'avg_reward_std': avg_reward_std,
        'avg_true_return': avg_true_return_mean,
        'avg_true_return_std': avg_true_return_std,
        'episode_lengths': episode_lengths
    }


if __name__ == "__main__":
    # Configuration (similar to train_rl.py)
    SEED = 100
    dataset = "butterfly_scB"
    obs_mode = "option3"
    action_gap = 1
    delta_actions = False
    norm_obs = False
    norm_reward = True
    randomize = True
    n_stack = 5  # Number of frames to stack (1 = no stacking, 4 = stack 4 frames)

    # Train SB3 agent (PPO by default, can be changed to "sac", "td3", etc.)
    algo = "ppo"  # Can be changed to "sac", "td3", "a2c", "ddpg"
    model, run_name = train_sb3_agent(
        algo=algo,
        dataset=dataset,
        total_timesteps=200_000,
        learning_rate=3e-4,
        obs_mode=obs_mode,
        action_gap=action_gap,
        delta_actions=delta_actions,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        randomize=randomize,
        seed=SEED,
        eval_freq=10_000,
        num_val_episodes=5,
        use_wandb=True,
        n_stack=n_stack,
        # Algorithm-specific hyperparameters (for PPO)
        n_steps=1024,
        batch_size=64,
    )

    # Optionally evaluate (similar to evaluate_and_visualize.py)
    # Use same evaluation settings as train_rl.py for fair comparison
    EVAL_SEED = 42  # Same as train_rl.py
    eval_randomize = True  # Same as train_rl.py (randomized=True)
    eval_num_runs = 10  # Same as train_rl.py (num_runs=10)

    best_model_path = os.path.join("checkpoints/rl_models_sb3", f"sb3_{algo}_agents_{dataset}", "best_model", "best_model.zip")
    vec_normalize_path = os.path.join("checkpoints/rl_models_sb3", f"sb3_{algo}_agents_{dataset}", "vec_normalize.pkl")
    if os.path.exists(best_model_path):
        eval_results = evaluate_model(
            best_model_path,
            dataset=dataset,
            obs_mode=obs_mode,
            action_gap=action_gap,
            delta_actions=delta_actions,
            norm_obs=norm_obs,  # Use same norm_obs as training
            seed=EVAL_SEED,  # Same seed as train_rl.py for fair comparison
            randomize=eval_randomize,  # Same as train_rl.py for fair comparison
            n_episodes=eval_num_runs,
            algo=algo,  # Pass algorithm name
            save_dir=str(project_root / "outputs" / "rl_training" / dataset / f"sb3_{algo}"),
            vec_normalize_path=vec_normalize_path if os.path.exists(vec_normalize_path) else None,
            n_stack=n_stack  # Use same n_stack as training
        )

        print(f"\nEvaluation Summary:")
        print(f"  Average true return: {eval_results['avg_true_return']:.3f} ± {eval_results['avg_true_return_std']:.3f}")
        print(f"  Results saved to: outputs/rl_training/{dataset}/sb3_{algo}")
        print(f"\nYou can now use evaluate_and_visualize.py to compare with other algorithms:")
        print(f"  python rl/evaluate_and_visualize.py --dataset {dataset} --run-test --randomize --evaluate --algorithms sb3_{algo} ppo sac")

        # Render a sample run
        env = PedNetParallelEnv(
            dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate", action_gap=action_gap
        )
        env.render(
            simulation_dir=str(project_root / f"outputs/rl_training/{dataset}/sb3_{algo}_run1"),
            variable='density',
            vis_actions=True,
            save_dir=None
        )
    else:
        print(f"\n⚠️  Warning: Best model not found at {best_model_path}")
        print("   Skipping evaluation. Model may not have been saved during training.")


