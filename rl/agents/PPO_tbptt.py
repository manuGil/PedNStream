# -*- coding: utf-8 -*-
# @Time    : 02/02/2026 11:00
# @Author  : mmai
# @FileName: PPO_tbptt.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from rl.rl_utils import compute_gae, save_with_best_return, validate_and_save_best, layer_init
import math
import os
import random
import collections
from .SAC import MLPEncoder, StackedEncoder
from .PPO_backup import AttentionPolicy, AttentionValueNetwork, train_on_policy_multi_agent

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PPOAgent:
    """PPO implementation for continuous action spaces with stateful LSTM policy."""

    def __init__(self, obs_dim, act_dim, act_low, act_high, actor_lr=3e-4, critic_lr=6e-4,
                 gamma=0.99, lmbda=0.95, epochs=10, device="cpu",
                 clip_eps=0.2, entropy_coef=0.01, entropy_coef_decay=0.995,
                 entropy_coef_min=0.001, kl_tolerance=0.01,
                 use_delta_actions=False, max_delta=2.5,
                 lstm_hidden_size=64, num_lstm_layers=1,
                 use_param_noise=False, param_noise_std=0.1,
                 param_noise_std_min=0.01,
                 use_action_noise=False, action_noise_std=0.1,
                 action_noise_std_min=0.01, num_episodes=100, tm_window=50,
                 use_lr_decay=False, lr_warmup_frac=0.05, lr_min_ratio=0.01):
        """
        Initialize PPO agent with LSTM, stacked observation, or GAT-LSTM networks.

        Args:
            obs_dim: Observation space dimension
            act_dim: Action space dimension
            act_low: Lower bound of action space
            act_high: Upper bound of action space
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            lmbda: GAE lambda parameter
            epochs: Number of epochs for PPO update
            clip_eps: PPO clipping parameter
            entropy_coef: Initial entropy bonus coefficient
            entropy_coef_decay: Exponential decay factor per update (default 0.995)
            entropy_coef_min: Minimum entropy coefficient (default 0.0001)
            kl_tolerance: KL divergence tolerance for early stopping
            use_delta_actions: If True, agent outputs delta actions
            max_delta: Maximum delta per step (only used if use_delta_actions=True)
            lstm_hidden_size: Hidden size for LSTM layers
            num_lstm_layers: Number of LSTM layers
            hidden_size: Hidden size for stacked networks (only used if use_stacked_obs=True)
            use_param_noise: If True, apply parameter noise to actor for exploration
            param_noise_std: Initial standard deviation for parameter noise
            param_noise_std_min: Minimum param_noise_std (default 0.01)
            use_action_noise: If True, apply action noise for exploration (default True)
            action_noise_std: Initial standard deviation for action noise (default 0.1)
            action_noise_std_min: Minimum action_noise_std (default 0.01)
            tm_window: Time window for truncated backpropagation through time (default 50)
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_low = torch.tensor(act_low, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, dtype=torch.float32)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.tm_window = tm_window # time window for truncated backpropagation through time
        self.tm_step = 1 # current step in the time window

        # Entropy coefficient with exponential decay
        self.entropy_coef_initial = entropy_coef
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.entropy_coef_min = entropy_coef_min
        self.update_count = 0  # Track number of updates for decay

        self.kl_tolerance = kl_tolerance
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

        # Delta action settings
        self.use_delta_actions = use_delta_actions
        self.max_delta = max_delta


        # LSTM settings
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        # Hidden states for actor and critic (reset at episode start)
        self.actor_hidden = None
        self.critic_hidden = None
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        # Create LSTM networks
        # self.actor = UDLSTMPolicyNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
        #                               num_layers=num_lstm_layers)
        # self.value_net = UDLSTMValueNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
        #                                  num_layers=num_lstm_layers)
        # self.actor = LSTMPolicyNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
        #                               num_layers=num_lstm_layers)
        # self.value_net = LSTMValueNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
        #                                num_layers=num_lstm_layers)
        self.actor = AttentionPolicy(obs_dim, act_dim, hidden_size=lstm_hidden_size,
                                        num_layers=num_lstm_layers)
        self.value_net = AttentionValueNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
                                                num_layers=num_lstm_layers)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=critic_lr)
        self.device = device

        # Parameter noise settings for exploration (linear decay based on update step)
        self.use_param_noise = use_param_noise
        self.param_noise_std_initial = param_noise_std
        self.param_noise_std = param_noise_std
        self.param_noise_std_min = param_noise_std_min
        self._param_noise_applied = False  # Track if noise is currently applied
        self._original_actor_params = None  # Store original params when noise is applied

        # Action noise settings for exploration (linear decay based on update step)
        self.use_action_noise = use_action_noise
        self.action_noise_std_initial = action_noise_std
        self.action_noise_std = action_noise_std
        self.action_noise_std_min = action_noise_std_min
        self.total_updates = num_episodes * 0.8

        # Learning rate scheduler settings
        self.use_lr_decay = use_lr_decay
        self.lr_warmup_frac = lr_warmup_frac
        self.lr_min_ratio = lr_min_ratio
        if self.use_lr_decay:
            self._setup_lr_scheduler()

    def reset_buffer(self):
        """Clear rollout buffer, reset LSTM hidden states, and apply parameter noise for new episode."""
        self.transition_dict = {
            'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],
            'actor_hiddens': [], 'model_hiddens': [],
            'true_rewards': []  # True (unnormalized) rewards for Dyna model training
        }
        # Reset LSTM hidden states (only if using LSTM or GAT-LSTM, will be initialized in first forward pass)
        self.actor_hidden = None
        self.critic_hidden = None
        self.tm_step = 1
        # Apply parameter noise at the start of each episode
        if self.use_param_noise:
            self._apply_param_noise()

    def init_batch_buffer(self):
        """Initialize batch buffer for storing multiple trajectories."""
        self.batch_buffer = []  # List of trajectory dicts
    
    def store_trajectory(self):
        """Store current trajectory to batch buffer and reset for next episode.
        
        Call this at the end of each episode before reset_buffer().
        """
        if not hasattr(self, 'batch_buffer'):
            self.init_batch_buffer()
        
        # Only store if trajectory has data
        if len(self.transition_dict['states']) > 0:
            # Deep copy the current trajectory
            trajectory = {
                'states': np.array(self.transition_dict['states']),
                'actions': np.array(self.transition_dict['actions']),
                'next_states': np.array(self.transition_dict['next_states']),
                'rewards': np.array(self.transition_dict['rewards']),
                'dones': np.array(self.transition_dict['dones']),
            }
            self.batch_buffer.append(trajectory)
    
    def clear_batch_buffer(self):
        """Clear the batch buffer after update."""
        self.batch_buffer = []
    
    def get_batch_size(self):
        """Return number of trajectories in batch buffer."""
        if not hasattr(self, 'batch_buffer'):
            return 0
        return len(self.batch_buffer)

    
    def reset_optimizer(self):
        """Reset optimizer state (keeps network weights, clears momentum). Used for curriculum learning."""
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.critic_lr)


    def _apply_param_noise(self):
        """Apply Gaussian noise to actor network parameters for exploration."""
        # First restore original params if noise was previously applied
        if self._param_noise_applied:
            self._restore_actor_params()
        
        # Store original parameters
        self._original_actor_params = {
            name: param.data.clone() for name, param in self.actor.named_parameters()
        }
        
        # Add noise to each parameter
        with torch.no_grad():
            for name, param in self.actor.named_parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * self.param_noise_std
                    param.data.add_(noise)
        
        self._param_noise_applied = True

    def _restore_actor_params(self):
        """Restore original actor parameters (remove noise)."""
        if self._original_actor_params is not None:
            with torch.no_grad():
                for name, param in self.actor.named_parameters():
                    if name in self._original_actor_params:
                        param.data.copy_(self._original_actor_params[name])
            self._original_actor_params = None
        self._param_noise_applied = False

    def _decay_param_noise_std(self):
        """Apply linear decay to parameter noise based on update step.
        
        Noise decays linearly from initial value to min value over total_updates.
        At update_count == total_updates, noise reaches param_noise_std_min.
        """
        progress = min(self.update_count / self.total_updates, 1.0)
        self.param_noise_std = self.param_noise_std_initial + \
            (self.param_noise_std_min - self.param_noise_std_initial) * progress

    def _decay_action_noise_std(self):
        """Apply linear decay to action noise based on update step.
        
        Noise decays linearly from initial value to min value over total_updates.
        At update_count == total_updates, noise reaches action_noise_std_min.
        """
        progress = min(self.update_count / self.total_updates, 1.0)
        self.action_noise_std = self.action_noise_std_initial + \
            (self.action_noise_std_min - self.action_noise_std_initial) * progress

    def store_transition(self, state, action, next_state, reward, done, true_reward=None):
        """Store transition in buffer, including hidden states for Dyna rollouts.
        
        Args:
            state: Current observation
            action: Action taken
            next_state: Next observation
            reward: Reward received (may be normalized)
            done: Terminal flag
            true_reward: True (unnormalized) reward for Dyna model training
        """
        self.transition_dict['states'].append(state)
        self.transition_dict['actions'].append(action)
        self.transition_dict['next_states'].append(next_state)
        self.transition_dict['rewards'].append(reward)
        self.transition_dict['dones'].append(done)
        # Store true reward for Dyna model training (use reward if true_reward not provided)
        self.transition_dict['true_rewards'].append(true_reward if true_reward is not None else reward)

    def take_action(self, state, deterministic: bool = False):
        """
        Take action given state using LSTM, stacked, or GAT-LSTM networks.

        Args:
            state: Observation array (single observation) or stacked observations (stack_size, obs_dim)
            deterministic: If True, use mean action (no sampling). Useful for evaluation.

        Returns:
            action: Action array of shape (act_dim,)
        """
        # Convert state to tensor
        state_array = np.array(state)
        state_tensor = torch.tensor(state_array, dtype=torch.float).unsqueeze(0).to(self.device)

        # # Forward pass through actor
        # if self.tm_step % self.tm_window == 0 and self.actor_hidden is not None: # truncated backpropagation through time
        #     self.actor_hidden = (self.actor_hidden[0].detach(), self.actor_hidden[1].detach())

        mu, sigma, self.actor_hidden = self.actor(state_tensor, self.actor_hidden)

        self.tm_step += 1

        if deterministic:
            # Use mean action for evaluation
            action = mu
        else:
            # Sample from distribution for exploration during training
            action_dist = torch.distributions.Normal(mu, sigma)
            action = action_dist.sample()
            
            # Add action noise for additional exploration (only during training)
            if self.use_action_noise:
                action_noise = torch.randn_like(action) * self.action_noise_std
                action = action + action_noise

        if self.use_delta_actions:
            # Agent outputs delta in [-max_delta, +max_delta]
            delta = torch.clamp(action, -self.max_delta, self.max_delta)
            action_out = delta.cpu().detach().numpy().squeeze()
        else:
            # Direct absolute action
            action = torch.clamp(action, self.act_low, self.act_high)
            action_out = action.cpu().detach().numpy().squeeze()
        
        return action_out

    def update(self):
        """Update policy and value networks using collected trajectory."""
        # Restore original actor parameters before update (remove noise)
        if self.use_param_noise and self._param_noise_applied:
            self._restore_actor_params()
        
        # Convert trajectory to tensors
        # For LSTM: states will be (1, T, obs_dim)
        states = torch.tensor(np.array(self.transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(self.transition_dict['actions'])).view(-1, self.act_dim).to(
            self.device)  # (T, act_dim)
        rewards = torch.tensor(np.array(self.transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)  # (T, 1)
        next_states = torch.tensor(np.array(self.transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(self.transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)  # (T, 1)

        # Prepare sequences for network forward pass
        states_seq = states.unsqueeze(0)  # (1, T, obs_dim)
        next_states_seq = next_states.unsqueeze(0)  # (1, T, obs_dim)

        # Compute targets with no_grad - these are constants for the update
        # Use TBPTT chunking to match the training forward pass hidden state trajectory
        T_total = states_seq.size(1)
        with torch.no_grad():
            all_next_values = []
            all_current_values = []
            all_old_log_probs = []
            
            actor_hidden_pre = None
            critic_hidden_s = None   # for states
            critic_hidden_ns = None  # for next_states
            
            for t in range(0, T_total, self.tm_window):
                end_t = min(t + self.tm_window, T_total)
                states_chunk = states_seq[:, t:end_t, :]
                next_states_chunk = next_states_seq[:, t:end_t, :]
                actions_chunk = actions[t:end_t]
                
                # Detach hidden states at chunk boundaries (same as training)
                if actor_hidden_pre is not None:
                    actor_hidden_pre = (actor_hidden_pre[0].detach(), actor_hidden_pre[1].detach())
                if critic_hidden_s is not None:
                    critic_hidden_s = (critic_hidden_s[0].detach(), critic_hidden_s[1].detach())
                if critic_hidden_ns is not None:
                    critic_hidden_ns = (critic_hidden_ns[0].detach(), critic_hidden_ns[1].detach())
                
                # Value network forward (states and next_states)
                chunk_current_values, critic_hidden_s = self.value_net(states_chunk, critic_hidden_s)
                chunk_current_values = chunk_current_values.squeeze(0)
                
                chunk_next_values, critic_hidden_ns = self.value_net(next_states_chunk, critic_hidden_ns)
                chunk_next_values = chunk_next_values.squeeze(0)
                
                # Actor forward for old log probs
                mu, std, actor_hidden_pre = self.actor(states_chunk, actor_hidden_pre)
                mu = mu.squeeze(0)
                std = std.squeeze(0)
                action_dist = torch.distributions.Normal(mu, std)
                chunk_old_log_probs = action_dist.log_prob(actions_chunk)
                
                all_current_values.append(chunk_current_values)
                all_next_values.append(chunk_next_values)
                all_old_log_probs.append(chunk_old_log_probs)
            
            # Concatenate chunks
            current_values = torch.cat(all_current_values, dim=0)  # (T, 1)
            next_values = torch.cat(all_next_values, dim=0)        # (T, 1)
            old_log_probs = torch.cat(all_old_log_probs, dim=0)    # (T, act_dim)

            td_target = rewards + self.gamma * next_values * (1 - dones)
            td_delta = td_target - current_values

            # Use shared compute_gae function
            advantage = compute_gae(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

            # Normalize advantage (crucial for stable training)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # PPO update epochs with truncated backpropagation through time
        for _ in range(self.epochs):
            # Forward pass through actor
            actor_hidden = None
            critic_hidden = None
            # Accumulate gradients across TBPTT chunks and apply a single optimizer step per epoch.
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            for t in range(0, T_total, self.tm_window):
                end_t = min(t + self.tm_window, T_total)
                chunk_len = end_t - t
                # Weight chunk losses so each timestep contributes equally across the full sequence.
                chunk_weight = float(chunk_len) / float(T_total)
                states_chunk = states_seq[:, t:end_t, :]
                actions_chunk = actions[t:end_t]
                old_log_probs_chunk = old_log_probs[t:end_t]
                advantage_chunk = advantage[t:end_t]
                td_target_chunk = td_target[t:end_t]
                
                if actor_hidden is not None:
                    actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
                if critic_hidden is not None:
                    critic_hidden = (critic_hidden[0].detach(), critic_hidden[1].detach())

                mu, std, actor_hidden = self.actor(states_chunk, actor_hidden)
                mu = mu.squeeze(0)
                std = std.squeeze(0)
                action_dist = torch.distributions.Normal(mu, std)
                entropy = action_dist.entropy().mean()
                log_probs = action_dist.log_prob(actions_chunk)
                log_ratio = (log_probs - old_log_probs_chunk).clamp(-20, 20)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantage_chunk
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage_chunk
                actor_loss = (torch.mean(-torch.min(surr1, surr2)) - self.entropy_coef * entropy) * chunk_weight
                # forward pass through critic
                current_values, critic_hidden = self.value_net(states_chunk, critic_hidden)
                current_values = current_values.squeeze(0)
                critic_loss = torch.mean(F.mse_loss(current_values, td_target_chunk)) * chunk_weight
                
                actor_loss.backward()
                critic_loss.backward()
    
                #     KL early stopping
                with torch.no_grad():
                    approx_kl = (log_probs - old_log_probs_chunk).mean()
                    if approx_kl > 1.5 * self.kl_tolerance:
                        break

            # gradient clipping (once per epoch, after accumulating all chunks)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            # single update step per epoch
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # Decay entropy coefficient after update
        self.update_count += 1
        self._decay_entropy_coef()
        self._step_lr_scheduler()
        
        # Decay parameter noise std after update
        if self.use_param_noise:
            self._decay_param_noise_std()
        
        # Decay action noise std after update
        if self.use_action_noise:
            self._decay_action_noise_std()

    def update_batch(self):
        """Update policy and value networks using multiple trajectories from batch buffer.
        
        This method processes each trajectory independently for TBPTT while accumulating
        gradients across all trajectories. Each trajectory maintains its own hidden state
        sequence to preserve temporal dependencies.
        
        Call store_trajectory() after each episode, then call update_batch() when
        enough trajectories are collected.
        """
        if not hasattr(self, 'batch_buffer') or len(self.batch_buffer) == 0:
            print("Warning: No trajectories in batch buffer. Skipping update.")
            return
        
        # Restore original actor parameters before update (remove noise)
        if self.use_param_noise and self._param_noise_applied:
            self._restore_actor_params()
        
        num_trajectories = len(self.batch_buffer)
        
        # Precompute targets and old_log_probs for all trajectories (no_grad)
        # Use TBPTT chunking to match the training forward pass hidden state trajectory
        trajectory_data = []
        all_advantages = []  # For global normalization
        
        with torch.no_grad():
            for traj in self.batch_buffer:
                # Convert trajectory to tensors
                states = torch.tensor(traj['states'], dtype=torch.float).to(self.device)
                actions = torch.tensor(traj['actions']).view(-1, self.act_dim).to(self.device)
                rewards = torch.tensor(traj['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
                next_states = torch.tensor(traj['next_states'], dtype=torch.float).to(self.device)
                dones = torch.tensor(traj['dones'], dtype=torch.float).view(-1, 1).to(self.device)
                
                # Prepare sequences: (1, T, obs_dim)
                states_seq = states.unsqueeze(0)
                next_states_seq = next_states.unsqueeze(0)
                T_traj = states_seq.size(1)
                
                # Compute values and old log probs using TBPTT chunks
                # This matches the chunked forward pass in the training loop
                all_next_values = []
                all_current_values = []
                all_old_log_probs = []
                
                actor_hidden = None
                critic_hidden_s = None  # for states
                critic_hidden_ns = None  # for next_states
                
                for t in range(0, T_traj, self.tm_window):
                    end_t = min(t + self.tm_window, T_traj)
                    states_chunk = states_seq[:, t:end_t, :]
                    next_states_chunk = next_states_seq[:, t:end_t, :]
                    actions_chunk = actions[t:end_t]
                    
                    # Detach hidden states at chunk boundaries (same as training)
                    if actor_hidden is not None:
                        actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
                    if critic_hidden_s is not None:
                        critic_hidden_s = (critic_hidden_s[0].detach(), critic_hidden_s[1].detach())
                    if critic_hidden_ns is not None:
                        critic_hidden_ns = (critic_hidden_ns[0].detach(), critic_hidden_ns[1].detach())
                    
                    # Value network forward (states and next_states)
                    chunk_current_values, critic_hidden_s = self.value_net(states_chunk, critic_hidden_s)
                    chunk_current_values = chunk_current_values.squeeze(0)  # (chunk_len, 1)
                    
                    chunk_next_values, critic_hidden_ns = self.value_net(next_states_chunk, critic_hidden_ns)
                    chunk_next_values = chunk_next_values.squeeze(0)  # (chunk_len, 1)
                    
                    # Actor forward for old log probs
                    mu, std, actor_hidden = self.actor(states_chunk, actor_hidden)
                    mu = mu.squeeze(0)
                    std = std.squeeze(0)
                    action_dist = torch.distributions.Normal(mu, std)
                    chunk_old_log_probs = action_dist.log_prob(actions_chunk)
                    
                    all_current_values.append(chunk_current_values)
                    all_next_values.append(chunk_next_values)
                    all_old_log_probs.append(chunk_old_log_probs)
                
                # Concatenate chunks
                current_values = torch.cat(all_current_values, dim=0)  # (T, 1)
                next_values = torch.cat(all_next_values, dim=0)  # (T, 1)
                old_log_probs = torch.cat(all_old_log_probs, dim=0)  # (T, act_dim)
                
                # TD targets and advantages
                td_target = rewards + self.gamma * next_values * (1 - dones)
                td_delta = td_target - current_values
                advantage = compute_gae(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
                
                trajectory_data.append({
                    'states_seq': states_seq,
                    'actions': actions,
                    'td_target': td_target,
                    'advantage': advantage,
                    'old_log_probs': old_log_probs,
                    'T': T_traj
                })
                all_advantages.append(advantage)
            
            # Global advantage normalization across all trajectories
            all_adv_cat = torch.cat(all_advantages, dim=0)
            global_mean = all_adv_cat.mean()
            global_std = all_adv_cat.std() + 1e-8
            
            # Normalize advantages
            for data in trajectory_data:
                data['advantage'] = (data['advantage'] - global_mean) / global_std
        
        # Total timesteps for weighting
        total_timesteps = sum(data['T'] for data in trajectory_data)
        
        # PPO update epochs
        for epoch in range(self.epochs):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            epoch_kl_exceeded = False
            
            # Process each trajectory independently (preserves hidden state continuity)
            for traj_idx, data in enumerate(trajectory_data):
                states_seq = data['states_seq']
                actions = data['actions']
                td_target = data['td_target']
                advantage = data['advantage']
                old_log_probs = data['old_log_probs']
                T_traj = data['T']
                
                # Weight for this trajectory (proportional to its length)
                traj_weight = float(T_traj) / float(total_timesteps)
                
                # TBPTT within this trajectory
                actor_hidden = None
                critic_hidden = None
                
                for t in range(0, T_traj, self.tm_window):
                    end_t = min(t + self.tm_window, T_traj)
                    chunk_len = end_t - t
                    # Chunk weight within trajectory
                    chunk_weight = float(chunk_len) / float(T_traj) * traj_weight
                    
                    # Extract chunks
                    states_chunk = states_seq[:, t:end_t, :]
                    actions_chunk = actions[t:end_t]
                    old_log_probs_chunk = old_log_probs[t:end_t]
                    advantage_chunk = advantage[t:end_t]
                    td_target_chunk = td_target[t:end_t]
                    
                    # Detach hidden states for TBPTT
                    if actor_hidden is not None:
                        actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
                    if critic_hidden is not None:
                        critic_hidden = (critic_hidden[0].detach(), critic_hidden[1].detach())
                    
                    # Actor forward pass
                    mu, std, actor_hidden = self.actor(states_chunk, actor_hidden)
                    mu = mu.squeeze(0)
                    std = std.squeeze(0)
                    action_dist = torch.distributions.Normal(mu, std)
                    entropy = action_dist.entropy().mean()
                    log_probs = action_dist.log_prob(actions_chunk)
                    
                    # PPO loss
                    log_ratio = (log_probs - old_log_probs_chunk).clamp(-20, 20)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * advantage_chunk
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage_chunk
                    actor_loss = (torch.mean(-torch.min(surr1, surr2)) - self.entropy_coef * entropy) * chunk_weight
                    
                    # Critic forward pass
                    current_values, critic_hidden = self.value_net(states_chunk, critic_hidden)
                    current_values = current_values.squeeze(0)
                    critic_loss = torch.mean(F.mse_loss(current_values, td_target_chunk)) * chunk_weight
                    
                    # Accumulate gradients
                    actor_loss.backward()
                    critic_loss.backward()
                    
                    # KL early stopping check
                    with torch.no_grad():
                        approx_kl = (log_probs - old_log_probs_chunk).mean()
                        if approx_kl > 1.5 * self.kl_tolerance:
                            epoch_kl_exceeded = True
                            break
                
                if epoch_kl_exceeded:
                    break
            
            # Gradient clipping and optimizer step (once per epoch)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            if epoch_kl_exceeded:
                break
        
        # Clear batch buffer after update
        self.clear_batch_buffer()
        
        # Decay coefficients
        self.update_count += 1
        self._decay_entropy_coef()
        self._step_lr_scheduler()
        if self.use_param_noise:
            self._decay_param_noise_std()
        if self.use_action_noise:
            self._decay_action_noise_std()

    def _decay_entropy_coef(self):
        """Apply exponential decay to entropy coefficient."""
        progress = min(self.update_count / self.total_updates, 1.0)
        self.entropy_coef = self.entropy_coef_initial + \
            (self.entropy_coef_min - self.entropy_coef_initial) * progress

    def _setup_lr_scheduler(self):
        """Set up linear warmup + cosine decay LR scheduler for actor and critic.

        Schedule:
          - Warmup phase (0 -> warmup_steps): linearly ramp from
            lr_min_ratio * base_lr -> base_lr.
          - Cosine decay phase (warmup_steps -> total_updates): cosine anneal
            from base_lr -> lr_min_ratio * base_lr.

        Uses torch.optim.lr_scheduler.LambdaLR so the scheduler state is
        automatically saved/loaded with optimizer checkpoints.
        """
        warmup_steps = max(1, int(self.total_updates * self.lr_warmup_frac))
        total = max(warmup_steps + 1, int(self.total_updates))  # avoid /0
        min_ratio = self.lr_min_ratio

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup from min_ratio -> 1.0
                return min_ratio + (1.0 - min_ratio) * (step / warmup_steps)
            else:
                # Cosine decay from 1.0 -> min_ratio
                decay_steps = total - warmup_steps
                progress = min((step - warmup_steps) / decay_steps, 1.0)
                return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.actor_optimizer, lr_lambda
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer, lr_lambda
        )

    def _step_lr_scheduler(self):
        """Advance LR scheduler by one step (call once per PPO update)."""
        if not self.use_lr_decay:
            return
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def get_current_lr(self):
        """Return current learning rates for logging."""
        if self.use_lr_decay:
            return {
                'actor_lr': self.actor_scheduler.get_last_lr()[0],
                'critic_lr': self.critic_scheduler.get_last_lr()[0],
            }
        return {
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
        }


    def get_config(self) -> dict:
        """Get agent configuration for saving/loading."""
        config = {
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'act_low': self.act_low.tolist(),
            'act_high': self.act_high.tolist(),
            'gamma': self.gamma,
            'lmbda': self.lmbda,
            'epochs': self.epochs,
            'clip_eps': self.clip_eps,
            'entropy_coef': self.entropy_coef_initial,  # Save initial value
            'entropy_coef_decay': self.entropy_coef_decay,
            'entropy_coef_min': self.entropy_coef_min,
            'use_delta_actions': self.use_delta_actions,
            'max_delta': self.max_delta,
            'use_param_noise': self.use_param_noise,
            'param_noise_std': self.param_noise_std_initial,
            'param_noise_std_min': self.param_noise_std_min,
            'use_action_noise': self.use_action_noise,
            'action_noise_std': self.action_noise_std_initial,
            'action_noise_std_min': self.action_noise_std_min,
            'total_updates': self.total_updates,
            'use_lr_decay': self.use_lr_decay,
            'lr_warmup_frac': self.lr_warmup_frac,
            'lr_min_ratio': self.lr_min_ratio,
        }


        config.update({
            'lstm_hidden_size': self.lstm_hidden_size,
            'num_lstm_layers': self.num_lstm_layers,
        })

        return config

    def save(self, path: str):
        """Save agent model parameters and training state."""
        # Ensure we save original (non-noisy) parameters
        if self.use_param_noise and self._param_noise_applied:
            self._restore_actor_params()
        
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.value_net.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.get_config(),
            'update_count': self.update_count,
            'current_entropy_coef': self.entropy_coef,
            'current_param_noise_std': self.param_noise_std if self.use_param_noise else None,
            'current_action_noise_std': self.action_noise_std if self.use_action_noise else None,
        }
        
        torch.save(save_dict, path)

    def load(self, path: str):
        """Load agent model parameters and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.value_net.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Restore entropy decay state if available (for continuing training)
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        if 'current_entropy_coef' in checkpoint:
            self.entropy_coef = checkpoint['current_entropy_coef']
        # Restore param noise state if available
        if 'current_param_noise_std' in checkpoint and checkpoint['current_param_noise_std'] is not None:
            self.param_noise_std = checkpoint['current_param_noise_std']
        # Restore action noise state if available
        if 'current_action_noise_std' in checkpoint and checkpoint['current_action_noise_std'] is not None:
            self.action_noise_std = checkpoint['current_action_noise_std']


def train_on_policy_multi_agent_batch(env, agents, delta_actions=False, num_episodes=50,
                                      num_trajectories_per_update=4, randomize=False,
                                      agents_saved_dir: str = None, use_wandb: bool = True,
                                      val_freq: int = 10, num_val_episodes: int = 3):
    """
    Train multiple on-policy agents (PPO) using batch trajectory updates.
    
    This function collects multiple trajectories before performing a single PPO update,
    which provides more stable gradient estimates compared to single-trajectory updates.

    Args:
        env: PettingZoo ParallelEnv
        agents: Dict mapping agent_id -> PPOAgent (must have update_batch method)
        delta_actions: If True, agents output delta actions
        num_episodes: Total number of episodes to train
        num_trajectories_per_update: Number of trajectories to collect before each update (default: 4)
        randomize: If True, randomize environment at reset
        agents_saved_dir: Directory to save agent checkpoints
        use_wandb: If True, log metrics to wandb (default: True)
        val_freq: Validation frequency - run validation every N updates (default: 10)
        num_val_episodes: Number of validation episodes to run (default: 3)

    Returns:
        return_dict: Dict mapping agent_id -> list of episode returns
        final_returns: Dict mapping agent_id -> final episode return
    """
    # Initialize wandb if available and not already initialized
    if use_wandb and WANDB_AVAILABLE:
        if wandb.run is None:
            wandb.init(project="crowd-control-rl", name="ppo-batch-training")
    
    # Initialize return tracking for each agent
    return_dict = {agent_id: [] for agent_id in agents.keys()}
    global_episode = 0
    global_update = 0  # Track number of batch updates

    # Track best average return across all agents
    best_avg_return = float('-inf')

    # Check if any agent uses stacked observations
    first_agent_id = next(iter(agents))
    uses_stacked_obs = hasattr(agents[first_agent_id], 'stack_size')

    # Initialize state history queues for stacked observations
    if uses_stacked_obs:
        first_agent_id = next(iter(agents))
        stack_size = agents[first_agent_id].stack_size
        state_history_queue = {agent_id: collections.deque(maxlen=stack_size) for agent_id in agents.keys()}

    # Initialize batch buffers for all agents
    for agent in agents.values():
        agent.init_batch_buffer()

    # Adjust total_updates for batch training so entropy / noise decay is based on
    # the expected number of PPO updates (not number of episodes).
    #
    # Original schedule (single-trajectory): total_updates ~= num_episodes * 0.8
    # Batch schedule (num_trajectories_per_update episodes per update):
    #   effective_updates ~= (num_episodes / num_trajectories_per_update) * 0.8
    #
    # This keeps the decay curve roughly comparable while accounting for fewer
    # optimizer steps.
    first_agent = next(iter(agents.values()))
    if hasattr(first_agent, "total_updates"):
        effective_updates = max(
            1,
            int(num_episodes / float(max(1, num_trajectories_per_update)) * 1),
        )
        for agent in agents.values():
            agent.total_updates = effective_updates

    # Track returns for current batch (for logging)
    batch_returns = {agent_id: [] for agent_id in agents.keys()}
    batch_true_returns = {agent_id: [] for agent_id in agents.keys()}

    num_iterations = 10
    episodes_per_iteration = num_episodes // num_iterations

    for i in range(num_iterations):
        with tqdm(total=episodes_per_iteration, desc='Iteration %d' % i) as pbar:
            for i_episode in range(episodes_per_iteration):
                # Reset buffers for all agents at start of episode
                for agent in agents.values():
                    agent.reset_buffer()

                # Reset environment
                if global_episode == 0:
                    obs, infos = env.reset(options={'randomize': False})
                else:
                    obs, infos = env.reset(options={'randomize': randomize})

                # Initialize state history queues for stacked observations
                state_stack = {}
                if uses_stacked_obs:
                    for agent_id in state_history_queue.keys():
                        state_history_queue[agent_id].clear()
                        for _ in range(agents[agent_id].stack_size):
                            state_history_queue[agent_id].append(obs[agent_id])
                        state_stack[agent_id] = np.array(state_history_queue[agent_id])

                episode_returns = {agent_id: 0.0 for agent_id in agents.keys()}
                episode_true_returns = {agent_id: 0.0 for agent_id in agents.keys()}
                done = False
                step = 0

                # Rollout episode
                while not done:
                    # Collect actions from all agents
                    actions = {}
                    absolute_actions = {}
                    
                    for agent_id, agent in agents.items():
                        if agent_id in state_stack:
                            agent_state = state_stack[agent_id]
                        else:
                            agent_state = obs[agent_id]
                        action = agent.take_action(agent_state)
                        
                        if delta_actions:
                            absolute_action = obs[agent_id].reshape(agents[agent_id].act_dim, -1)[:,-1] + action
                            absolute_action = np.clip(absolute_action, agents[agent_id].act_low, agents[agent_id].act_high)
                            absolute_actions[agent_id] = absolute_action
                        else:
                            absolute_actions[agent_id] = action
                        actions[agent_id] = action
                    
                    # Step environment
                    next_obs, rewards, terms, truncs, infos = env.step(absolute_actions)
                    next_state_stack = {}

                    # Store transitions for all agents
                    for agent_id, agent in agents.items():
                        if agent_id in state_stack:
                            state_history_queue[agent_id].append(next_obs[agent_id])
                            next_state_stack[agent_id] = np.array(state_history_queue[agent_id])
                            stored_state = state_stack[agent_id]
                            stored_next_state = next_state_stack[agent_id]
                        else:
                            stored_state = obs[agent_id]
                            stored_next_state = next_obs[agent_id]

                        agent.store_transition(
                            state=stored_state,
                            action=actions[agent_id],
                            next_state=stored_next_state,
                            reward=rewards[agent_id],
                            done=terms[agent_id],
                            true_reward=infos[agent_id].get('true_reward', rewards[agent_id])
                        )
                        episode_returns[agent_id] += rewards[agent_id]
                        
                        if agent_id in infos and 'true_reward' in infos[agent_id]:
                            episode_true_returns[agent_id] += infos[agent_id]['true_reward']
                        else:
                            episode_true_returns[agent_id] += rewards[agent_id]

                    obs = next_obs
                    if uses_stacked_obs:
                        state_stack = next_state_stack

                    step += 1
                    done = any(terms.values()) or any(truncs.values())

                # Episode finished - store trajectory to batch buffer
                for agent_id, agent in agents.items():
                    agent.store_trajectory()
                    return_dict[agent_id].append(episode_returns[agent_id])
                    batch_returns[agent_id].append(episode_returns[agent_id])
                    batch_true_returns[agent_id].append(episode_true_returns[agent_id])

                global_episode += 1

                # Check if we have enough trajectories for batch update
                first_agent = next(iter(agents.values()))
                if first_agent.get_batch_size() >= num_trajectories_per_update:
                    # Perform batch update for all agents
                    for agent_id, agent in agents.items():
                        if hasattr(env, 'ret_rms') and env.ret_rms is not None:
                            try:
                                agent.set_reward_normalizer_var(float(env.ret_rms.var))
                            except:
                                pass
                        agent.update_batch()
                    
                    global_update += 1

                    # Log to wandb after batch update
                    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                        log_dict = {
                            'update': global_update,
                            'episode': global_episode,
                            'batch_avg_normalized_return': np.mean([np.mean(batch_returns[aid]) for aid in agents.keys()]),
                            'batch_avg_true_return': np.mean([np.mean(batch_true_returns[aid]) for aid in agents.keys()]),
                            'trajectories_per_update': num_trajectories_per_update,
                            'episode_steps': step
                        }
                        for agent_id in agents.keys():
                            log_dict[f'agent_{agent_id}_batch_avg_return'] = np.mean(batch_returns[agent_id])
                            log_dict[f'agent_{agent_id}_batch_avg_true_return'] = np.mean(batch_true_returns[agent_id])
                        # Log LR schedule from first agent
                        first_agent_lr = first_agent.get_current_lr()
                        log_dict['actor_lr'] = first_agent_lr['actor_lr']
                        log_dict['critic_lr'] = first_agent_lr['critic_lr']
                        log_dict['entropy_coef'] = first_agent.entropy_coef
                        wandb.log(log_dict)

                    # Run validation and save best model
                    if agents_saved_dir and global_update > (num_episodes // num_trajectories_per_update) // 2 and global_update % val_freq == 0:
                        best_avg_return = validate_and_save_best(
                            env, agents, agents_saved_dir,
                            delta_actions=delta_actions,
                            num_val_episodes=num_val_episodes,
                            randomize=True,  # Use deterministic env for consistent model comparison
                            best_avg_return=best_avg_return,
                            global_episode=global_episode,
                            use_wandb=use_wandb and WANDB_AVAILABLE
                        )

                    # Reset batch return tracking
                    batch_returns = {agent_id: [] for agent_id in agents.keys()}
                    batch_true_returns = {agent_id: [] for agent_id in agents.keys()}

                # Update progress bar
                if (i_episode + 1) % 10 == 0:
                    avg_return = np.mean([np.mean(return_dict[aid][-10:]) for aid in agents.keys()])
                    avg_true_return = np.mean(list(episode_true_returns.values()))
                    pbar.set_postfix({
                        'episode': '%d' % global_episode,
                        'update': '%d' % global_update,
                        'norm_ret': '%.3f' % avg_return,
                        'true_ret': '%.3f' % avg_true_return,
                        'steps': step
                    })
                pbar.update(1)
                
                # Print episode rewards
                for agent_id in agents.keys():
                    print(f"Agent {agent_id} episode reward: {episode_returns[agent_id].item():.3f}")
                print(f"All agents episode reward: {sum(episode_returns.values()).item():.3f}")

    # Final returns
    final_returns = {agent_id: return_dict[agent_id][-1] if return_dict[agent_id] else 0.0 
                     for agent_id in agents.keys()}
    
    return return_dict, final_returns