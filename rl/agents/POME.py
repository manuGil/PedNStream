# -*- coding: utf-8 -*-
# @Time    : 04/02/2026 14:54
# @Author  : mmai
# @FileName: POME.py
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
from torch_geometric.nn import DenseGATConv
from torch_geometric.utils import to_dense_adj
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# =============================================================================
# Local Dynamic Prediction Model
# =============================================================================
class LocalDynamicModel(nn.Module):
    """
    Local Dynamic Prediction Model
    """
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.features_per_link = obs_dim // act_dim

        self.lstm = nn.LSTM(
            input_size=self.features_per_link,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.link_model = nn.Linear(hidden_size, hidden_size)
        
        # Upstream/Downstream aggregation model
        # Input: [link_features, other_links_sum] -> hidden_size
        self.ud_model = nn.Linear(2 * hidden_size, hidden_size)

        # Shared latent layer for action coordination
        # self.shared_latent_layer = nn.Linear(hidden_size * act_dim, hidden_size * act_dim)

        self.reward_head = nn.Linear(hidden_size + act_dim, 1) # R(s_t, a_t)
        self.state_head = nn.Linear(hidden_size + 1, self.features_per_link - 1) # P(s_t_next | s_t, a_t), s_t_next exclude the gate width

    def forward(self, x, a, hidden=None):
        if x.dim() == 3:
            x = x.squeeze()
        seq_len = x.shape[0]

        x_lstm_input = x.view(seq_len, self.act_dim, self.features_per_link).transpose(0, 1)
        lstm_out, hidden_out = self.lstm(x_lstm_input, hidden) # (seq_len, num_links, hidden_size)
        lstm_features = lstm_out.transpose(0, 1)
        link_features = self.link_model(lstm_features)
        all_links_sum = link_features.sum(dim=1)
        other_links_features = all_links_sum.unsqueeze(1) - link_features
        combined_features = torch.cat([link_features, other_links_features], dim=2)
        ud_features = self.ud_model(combined_features) # (seq_len, num_links, hidden_size)

        reward = self.reward_head(F.relu(torch.cat([ud_features.mean(dim=1), a], dim=1)))
        next_state = self.state_head(F.relu(torch.cat([ud_features, a.unsqueeze(2)], dim=2))) # (seq_len, num_links, features_per_link - 1)
        # concate a with next_state
        current_gate_width = x.view(seq_len, self.act_dim, self.features_per_link)[..., -1] + a
        next_state = torch.cat([next_state, current_gate_width.unsqueeze(2)], dim=2)
        return reward, next_state, hidden_out


class SimpleDynamicModel(nn.Module):
    """
    Simple LSTM + MLP Dynamic Prediction Model
    """
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.features_per_link = obs_dim // act_dim

        # LSTM processes the full observation sequence
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # MLP for reward prediction: R(s_t, a_t)
        self.reward_mlp = nn.Sequential(
            nn.Linear(hidden_size + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # MLP for next state prediction: P(s_t_next | s_t, a_t)
        # Predicts features_per_link - 1 for each link (excluding gate width)
        self.state_mlp = nn.Sequential(
            nn.Linear(hidden_size + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim * (self.features_per_link - 1))
        )

    def forward(self, x, a, hidden=None):
        if x.dim() == 3:
            x = x.squeeze()
        seq_len = x.shape[0]

        # LSTM input: (batch, seq_len, obs_dim) - treat each timestep as single obs
        x_lstm = x.unsqueeze(1)  # (seq_len, 1, obs_dim)
        lstm_out, hidden_out = self.lstm(x_lstm, hidden)
        lstm_features = lstm_out.squeeze(1)  # (seq_len, hidden_size)

        # Concatenate LSTM features with action
        combined = torch.cat([lstm_features, a], dim=1)

        # Predict reward
        reward = self.reward_mlp(combined)

        # Predict next state (excluding gate width)
        next_state_flat = self.state_mlp(combined)
        next_state = next_state_flat.view(seq_len, self.act_dim, self.features_per_link - 1)

        # Append gate width: current gate + action delta
        current_gate_width = x.view(seq_len, self.act_dim, self.features_per_link)[..., -1] + a
        next_state = torch.cat([next_state, current_gate_width.unsqueeze(2)], dim=2)

        return reward, next_state, hidden_out


class POMEAgent:
    """PPO implementation for continuous action spaces with stateful LSTM policy."""

    def __init__(self, obs_dim, act_dim, act_low, act_high, actor_lr=3e-4, critic_lr=6e-4,
                 gamma=0.99, lmbda=0.95, epochs=10, device="cpu",
                 clip_eps=0.2, entropy_coef=0.01, entropy_coef_decay=0.995,
                 entropy_coef_min=0.001, kl_tolerance=0.01,
                 use_delta_actions=False, max_delta=2.5,
                 lstm_hidden_size=64, num_lstm_layers=1,
                 hidden_size=64,
                 use_param_noise=False, param_noise_std=0.1,
                 param_noise_std_min=0.01,
                 use_action_noise=False, action_noise_std=0.1,
                 action_noise_std_min=0.01, num_episodes=100, tm_window=50,
                 model_lr=3e-4, alpha=0.1, alpha_min=0.0, norm_reward=False):
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
        self.model_lr = model_lr

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
        # Store learning rates for optimizer reset
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
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

        # POME alpha coefficient with linear decay
        self.alpha_initial = alpha
        self.alpha = alpha
        self.alpha_min = alpha_min

        # Reward normalization flag for POME
        self.norm_reward = norm_reward
        self.reward_ret_var = 1.0  # Default, updated via set_reward_normalizer_var()
        self.clip_reward = 10.0  # Match wrapper's default clip_reward

        # Dyna-PPO: Dynamics model for model-based augmentation

            # self.dynamic_model = SimpleDynamicModel(
            #     obs_dim, act_dim, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers
            # ).to(device)
        self.dynamic_model = LocalDynamicModel(
            obs_dim, act_dim, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers
        ).to(device)
        self.model_optimizer = torch.optim.Adam(self.dynamic_model.parameters(), lr=model_lr)
        # self.model_hidden = None  # Hidden state for dynamics model
        self.model_train_steps = 10  # Number of gradient steps per update


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
        # if self.use_dyna:
        # self.model_hidden = None
        self.tm_step = 1
        # Apply parameter noise at the start of each episode
        if self.use_param_noise:
            self._apply_param_noise()

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
        # Store hidden states for Dyna (deep copy to avoid reference issues)
        # if self.use_dyna:
        actor_h = None
        if self.actor_hidden is not None:
            actor_h = (self.actor_hidden[0].detach().clone(), self.actor_hidden[1].detach().clone())
        # model_h = None
        # if self.model_hidden is not None:
        #     model_h = (self.model_hidden[0].detach().clone(), self.model_hidden[1].detach().clone())
        self.transition_dict['actor_hiddens'].append(actor_h)
        # self.transition_dict['model_hiddens'].append(model_h)
    
    def set_reward_normalizer_var(self, ret_var: float):
        """Set reward normalizer variance from training wrapper.
        
        This is used to normalize simulated rewards to match real reward scale.
        Call this before update() to sync with the wrapper's running statistics.
        """
        #  if self.use_dyna:
        if True:
            self.reward_ret_var = max(ret_var, 1e-8)  # Prevent division by zero

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
        
        # Update dynamics model hidden state for Dyna (needed for storing)
        # if self.use_dyna:
        # with torch.no_grad():
        #     action_tensor = torch.tensor(action_out, dtype=torch.float).unsqueeze(0).to(self.device)
        #     _, _, self.model_hidden = self.dynamic_model(state_tensor, action_tensor, self.model_hidden)
        
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

        # train dynamics model
        true_rewards = torch.tensor(np.array(self.transition_dict['true_rewards']),
                                    dtype=torch.float).view(-1, 1).to(self.device)
        model_loss = self.train_dynamics_model(states, actions, next_states, true_rewards)

        # Compute targets with no_grad - these are constants for the update
        with torch.no_grad():
            # Process entire sequences through value network
            # LSTM version
            next_values, _ = self.value_net(next_states_seq)  # (1, T, 1)
            next_values = next_values.squeeze(0)  # (T, 1)

            current_values, _ = self.value_net(states_seq)  # (1, T, 1)
            current_values = current_values.squeeze(0)  # (T, 1)

            predicted_rewards, predicted_next_states, _ = self.dynamic_model(states, actions, hidden=None)
            # Normalize predicted rewards if norm_reward is enabled
            # Match the wrapper's approach: normalize by sqrt(return variance)
            if self.norm_reward:
                predicted_rewards = torch.clamp(
                    predicted_rewards / (self.reward_ret_var ** 0.5 + 1e-8),
                    -self.clip_reward, self.clip_reward
                )
            predicted_next_values, _ = self.value_net(predicted_next_states)
            Q_f = rewards + self.gamma * next_values * (1 - dones)
            Q_b = predicted_rewards + self.gamma * predicted_next_values * (1 - dones)
            epsilon = torch.abs(Q_f - Q_b)
            epsilon_median = torch.median(epsilon)
            td_target = rewards + self.gamma * next_values * (1 - dones)
            td_delta = td_target - current_values # actual td delta
            upper_bound = torch.abs(td_delta)
            lower_bound = -torch.abs(td_delta)
            td_delta_pome = td_delta + self.alpha * torch.clamp(epsilon - epsilon_median, lower_bound, upper_bound)

            # Use shared compute_gae function
            advantage = compute_gae(self.gamma, self.lmbda, td_delta_pome.cpu()).to(self.device)

            # Normalize advantage (crucial for stable training)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Process sequence through actor to get old log probs
            mu, std, _ = self.actor(states_seq)  # (1, T, act_dim)
            mu = mu.squeeze(0)  # (T, act_dim)
            std = std.squeeze(0)  # (T, act_dim)
            action_dist = torch.distributions.Normal(mu, std)
            old_log_probs = action_dist.log_prob(actions)

        # PPO update epochs with truncated backpropagation through time
        T_total = states_seq.size(1)
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
        self._decay_alpha()
        
        # Decay parameter noise std after update
        if self.use_param_noise:
            self._decay_param_noise_std()
        
        # Decay action noise std after update
        if self.use_action_noise:
            self._decay_action_noise_std()

        # =================================================================
        # Dyna-PPO: Model-based data augmentation
        # =================================================================
        # if self.use_dyna:
            # Step 1: Train dynamics model on real data using TRUE rewards
        
        # Track model loss for adaptive dreaming threshold
        # self.model_loss_history.append(model_loss)
        # if len(self.model_loss_history) > self.model_loss_window:
        #     self.model_loss_history.pop(0)
        
            # Log model loss and dreaming status if wandb is available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'dynamics_model_loss': model_loss,
            })

    def _decay_entropy_coef(self):
        """Apply linear decay to entropy coefficient."""
        progress = min(self.update_count / self.total_updates, 1.0)
        self.entropy_coef = self.entropy_coef_initial + \
            (self.entropy_coef_min - self.entropy_coef_initial) * progress

    def _decay_alpha(self):
        """Apply linear decay to POME alpha coefficient."""
        progress = min(self.update_count / self.total_updates, 1.0)
        self.alpha = self.alpha_initial + \
            (self.alpha_min - self.alpha_initial) * progress

    # =========================================================================
    # Dyna-PPO Methods
    # =========================================================================

    def train_dynamics_model(self, states, actions, next_states, true_rewards):
        """
        Train the dynamics model on real transitions using TRUE (unnormalized) rewards.
        
        Args:
            states: (T, obs_dim) tensor
            actions: (T, act_dim) tensor  
            next_states: (T, obs_dim) tensor
            true_rewards: (T, 1) tensor - TRUE rewards (not normalized)
            
        Returns:
            float: Average model loss over training steps
        """
        # if not self.use_dyna:
        #     return 0.0
        
        # Reshape states for model: (seq_len, obs_dim) -> add batch dim
        states_seq = states.unsqueeze(0)  # (1, T, obs_dim)
        actions_squeezed = actions.float()  # (T, act_dim)
        
        # Target: reshape next_states to match prediction shape
        features_per_link = self.obs_dim // self.act_dim
        target_next_states = next_states.view(-1, self.act_dim, features_per_link)  # (T, num_links, features_per_link)
        
        total_loss = 0.0
        
        # Multiple gradient steps for better convergence
        for _ in range(self.model_train_steps):
            self.model_optimizer.zero_grad()
            
            # Forward through dynamics model
            pred_rewards, pred_next_states, _ = self.dynamic_model(states_seq, actions_squeezed, hidden=None)
            # pred_rewards: (T, 1), pred_next_states: (T, num_links, features_per_link)
            
            # Compute losses - use TRUE rewards for stable training
            state_loss = F.mse_loss(pred_next_states, target_next_states)
            reward_loss = F.mse_loss(pred_rewards, true_rewards)
            model_loss = state_loss + reward_loss
            print(f"Model loss: {model_loss.item()}, State loss: {state_loss.item()}, Reward loss: {reward_loss.item()}")
            
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dynamic_model.parameters(), max_norm=0.5)
            self.model_optimizer.step()
            
            total_loss += model_loss.item()
        
        return total_loss / self.model_train_steps  # Return average loss


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
            'alpha': self.alpha_initial,
            'alpha_min': self.alpha_min,
        }


        config.update({
            'lstm_hidden_size': self.lstm_hidden_size,
            'num_lstm_layers': self.num_lstm_layers,
        })

        # Dyna-PPO settings
        config.update({
            'model_lr': self.model_lr,
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
            'current_alpha': self.alpha,
        }
        # Save dynamics model if using Dyna
        # if self.use_dyna:
        save_dict['dynamic_model_state_dict'] = self.dynamic_model.state_dict()
        save_dict['model_optimizer_state_dict'] = self.model_optimizer.state_dict()
        
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
        # Restore alpha state if available
        if 'current_alpha' in checkpoint:
            self.alpha = checkpoint['current_alpha']
        # Restore dynamics model if using Dyna
        # if self.use_dyna and 'dynamic_model_state_dict' in checkpoint:
        if 'dynamic_model_state_dict' in checkpoint:
            self.dynamic_model.load_state_dict(checkpoint['dynamic_model_state_dict'])
            if 'model_optimizer_state_dict' in checkpoint:
                self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])