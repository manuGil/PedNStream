# -*- coding: utf-8 -*-
"""
Train PPO agent using sequential curriculum learning across multiple scenarios.

This script trains a single agent through multiple scenarios (scA -> scB -> scC -> scD)
with weight transfer between stages. The goal is to learn a generalized policy.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
import torch
from rl import PedNetParallelEnv
from rl.rl_utils import (
    RunningNormalizeWrapper,
    train_sequential_curriculum,
    evaluate_agents,
    load_all_agents,
)
from rl.agents.PPO_backup import PPOAgent
from rl.agents.PPO_dyna import PPOAgent as PPOAgent_dyna
from rl.agents.POME import POMEAgent


def create_env_factory(obs_mode: str = "option3", normalize_obs: bool = False, 
                       norm_wrapper: bool = False, norm_reward: bool = True, action_gap: int = 1):
    """
    Create an environment factory function.
    
    Args:
        obs_mode: Observation mode for the environment
        normalize_obs: Whether to normalize observations in env builder
        norm_wrapper: Whether to wrap with RunningNormalizeWrapper
        norm_reward: Whether to normalize rewards in wrapper
        action_gap: Action gap for environment
        
    Returns:
        Callable that creates environment given dataset name
    """
    def factory(dataset: str):
        base_env = PedNetParallelEnv(
            dataset=dataset,
            normalize_obs=normalize_obs,
            obs_mode=obs_mode,
            render_mode="animate",
            action_gap=action_gap
        )
        if norm_wrapper:
            env = RunningNormalizeWrapper(base_env, norm_obs=False, norm_reward=norm_reward)
        else:
            env = base_env
        return env
    return factory


def create_agent(algo: str, obs_dim: int, act_dim: int, act_low, act_high, total_updates: int):
    """
    Create agent based on algorithm choice.
    
    Args:
        algo: Algorithm name ("ppo", "ppo_dyna", "pome")
        obs_dim: Observation dimension
        act_dim: Action dimension
        act_low: Action space lower bound
        act_high: Action space upper bound
        total_updates: Total number of updates for noise/entropy decay
        
    Returns:
        Agent instance
    """
    if algo == "ppo":
        return PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=act_low,
            act_high=act_high,
            actor_lr=9e-5,
            critic_lr=2e-4,
            gamma=0.99,
            lmbda=0.96,
            entropy_coef=0.01,
            kl_tolerance=0.01,
            use_delta_actions=True,
            max_delta=2.5,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            use_gat_lstm=False,
            gat_hidden_size=64,
            use_stacked_obs=False,
            stack_size=0,
            hidden_size=64,
            kernel_size=4,
            use_param_noise=False,
            use_action_noise=False,
            total_updates=total_updates,
            param_noise_std_min=0
        )
    
    elif algo == "ppo_dyna":
        return PPOAgent_dyna(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=act_low,
            act_high=act_high,
            actor_lr=9e-5,
            critic_lr=2e-4,
            gamma=0.99,
            lmbda=0.96,
            entropy_coef=0.01,
            kl_tolerance=0.01,
            use_delta_actions=True,
            max_delta=2.5,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            use_param_noise=False,
            use_action_noise=False,
            num_episodes=total_updates,
            tm_window=50,
            use_dyna=False,
            model_lr=3e-4,
            dream_rollouts=20,
            dream_horizon=5,
        )
    
    elif algo == "pome":
        return POMEAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=act_low,
            act_high=act_high,
            actor_lr=9e-5,
            critic_lr=2e-4,
            gamma=0.99,
            lmbda=0.96,
            entropy_coef=0.01,
            kl_tolerance=0.01,
            use_delta_actions=True,
            max_delta=2.5,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            use_param_noise=False,
            use_action_noise=False,
            num_episodes=total_updates,
            tm_window=50,
            model_lr=5e-4,
            norm_reward=True,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Supported: ppo, ppo_dyna, pome")


if __name__ == "__main__":
    # ==========================================================================
    # Configuration
    # ==========================================================================
    algo = "ppo_dyna"  # Options: "ppo", "ppo_dyna", "pome"
    SEED = 60
    STATE_OPTION = "option3"
    NORM_WRAPPER = True  # Whether to use RunningNormalizeWrapper
    NORM_REWARD = True
    ACTION_GAP = 1
    
    # Curriculum settings
    SCENARIO_SEQUENCE = [
        "butterfly_scA",
        "butterfly_scC",
        "butterfly_scB",
        "butterfly_scD",
    ]
    EPISODES_PER_STAGE = 200
    VAL_FREQ = 10
    NUM_VAL_EPISODES = 5
    #
    # # Set seeds
    # torch.manual_seed(SEED)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(SEED)
    # np.random.seed(SEED)
    #
    # print("=" * 60)
    # print(f"Sequential Curriculum Training - {algo.upper()}")
    # print(f"Scenarios: {' -> '.join(SCENARIO_SEQUENCE)}")
    # print(f"Episodes per stage: {EPISODES_PER_STAGE}")
    # print("=" * 60)
    #
    # # ==========================================================================
    # # Create environment factory and initial environment (to get dimensions)
    # # ==========================================================================
    env_factory = create_env_factory(
        obs_mode=STATE_OPTION,
        normalize_obs=False,
        norm_wrapper=NORM_WRAPPER,
        norm_reward=NORM_REWARD,
        action_gap=ACTION_GAP
    )
    #
    # # Create initial env to get observation/action dimensions
    # initial_env = env_factory(SCENARIO_SEQUENCE[0])
    # initial_env.seed(SEED)
    #
    # # Get the first (and only) agent ID - focusing on single agent
    # agent_id = list(initial_env.possible_agents)[0]
    # print(f"Training agent: {agent_id}")
    # print(f"Observation dim: {initial_env.observation_space(agent_id).shape[0]}")
    # print(f"Action dim: {initial_env.action_space(agent_id).shape[0]}")
    #
    # # ==========================================================================
    # # Create Agent based on algorithm choice
    # # ==========================================================================
    # total_updates = EPISODES_PER_STAGE * len(SCENARIO_SEQUENCE)
    # agent = create_agent(
    #     algo=algo,
    #     obs_dim=initial_env.observation_space(agent_id).shape[0],
    #     act_dim=initial_env.action_space(agent_id).shape[0],
    #     act_low=initial_env.action_space(agent_id).low,
    #     act_high=initial_env.action_space(agent_id).high,
    #     total_updates=total_updates,
    # )

    # ==========================================================================
    # Train with Curriculum
    # ==========================================================================
    save_dir = f"./checkpoints/curriculum_{algo}"
    # results = train_sequential_curriculum(
    #     agent=agent,
    #     scenario_sequence=SCENARIO_SEQUENCE,
    #     env_factory=env_factory,
    #     episodes_per_stage=EPISODES_PER_STAGE,
    #     delta_actions=True,
    #     randomize=True,
    #     val_freq=VAL_FREQ,
    #     num_val_episodes=NUM_VAL_EPISODES,
    #     save_dir=save_dir,
    #     use_wandb=True,
    # )

    # ==========================================================================
    # Print Summary
    # ==========================================================================
    # print("\n" + "=" * 60)
    # print("Training Summary")
    # print("=" * 60)
    # for scenario, returns in results['stage_returns'].items():
    #     avg_return = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
    #     print(f"  {scenario}: avg last 10 episodes = {avg_return:.3f}")
    #
    # print("\nFinal Validation Results:")
    # for scenario, val_return in results['final_validation'].items():
    #     print(f"  {scenario}: {val_return:.3f}")
    # print(f"  Average: {results['avg_all_scenarios']:.3f}")
    # print("=" * 60)

    # ==========================================================================
    # Optional: Evaluate final model with more runs
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Extended Evaluation (10 runs per scenario)")
    print("=" * 60)

    # Load final model
    agents_loaded, _ = load_all_agents(
        save_dir=f"{save_dir}/final",
        device="cpu"
    )
    
    # Get the loaded agent (there's only one)
    loaded_agent = list(agents_loaded.values())[0]
    
    for scenario in SCENARIO_SEQUENCE:
        eval_env = env_factory(scenario)
        # Get agent_id from this scenario's environment
        eval_agent_id = eval_env.possible_agents[0]
        eval_results = evaluate_agents(
            eval_env,
            {eval_agent_id: loaded_agent},
            delta_actions=True,
            deterministic=True,
            seed=42,
            randomize=True,
            num_runs=10,
            verbose=False
        )
        print(f"  {scenario}: {eval_results['avg_reward']:.3f} ± {eval_results['avg_reward_std']:.3f}")
    
    print("=" * 60)
