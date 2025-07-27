# File: training_loop.py (versione con correzione del bug di caricamento checkpoint)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.func import vmap
from typing import Callable, Optional, List
import logging
import time

from .critic_transformer import CriticTransformer
from .critic_transformer import CriticDecisionTransformer
from .parallel_env import ParallelEnvManager
from .actor import ActorMPC
from .Checkpoint_Manager import CheckpointManager


def create_history_buffers(flat_buffer: torch.Tensor, history_len: int) -> torch.Tensor:
    num_envs, episode_len, data_dim = flat_buffer.shape
    padding = torch.zeros(num_envs, history_len - 1, data_dim, device=flat_buffer.device, dtype=flat_buffer.dtype)
    padded_buffer = torch.cat([padding, flat_buffer], dim=1)
    sequences = padded_buffer.unfold(dimension=1, size=history_len, step=1)
    return sequences.permute(0, 1, 3, 2)


def compute_gae_and_returns(
        rewards: torch.Tensor, values: torch.Tensor, *, gamma: float = 0.99, lam: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    values_extended = torch.cat([values, values[-1:]], dim=0)
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_extended[t + 1] - values_extended[t]
        last_advantage = delta + gamma * lam * last_advantage
        advantages[t] = last_advantage
    return advantages + values, advantages


def train(
        env_fn: Callable, actor: ActorMPC, critic: nn.Module,
        reward_fn: Optional[Callable],
        history_len: int = 1,
        steps: int = 1000, *, num_envs: int = 24,
        episode_len: int = 250, mpc_horizon: int, ppo_epochs: int = 10,
        clip_param: float = 0.2, entropy_coeff: float = 0.01,
        mpve_coeff: float = 0.1, gamma: float = 0.99, lam: float = 0.97,
        checkpoint_manager: Optional[CheckpointManager] = None, resume_from: str = "latest",
        use_amp: bool = False, seed: int | None = None,
) -> List[float]:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    device = actor.device
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
    scaler = GradScaler(enabled=use_amp)

    start_step_idx = 0
    if checkpoint_manager:
        start_step_idx = checkpoint_manager.load(
            device=device, resume_from=resume_from, actor=actor, critic=critic,
            actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, scaler=scaler
        )

    envs = ParallelEnvManager(env_fn, num_envs, device)
    obs_dim = envs.single_observation_space_shape[0]

    batched_compute_gae = vmap(compute_gae_and_returns, in_dims=(0, 0))
    if reward_fn is not None:
        batched_reward_fn = vmap(reward_fn)
    else:
        batched_reward_fn = None

    training_rewards = []

    for step_idx in range(start_step_idx, steps):
        # ... (Fase 1 e 1.5 - Raccolta dati e creazione cronologia)
        step_start_time = time.time()
        logging.info(f"--- Inizio Step di Training {step_idx + 1}/{steps} (indice: {step_idx}) ---")
        logging.info(f"Fase 1: Raccolta di {episode_len} campioni da {num_envs} ambienti...")
        states_buf = torch.zeros(num_envs, episode_len, obs_dim, device=device, dtype=actor.dtype)
        actions_buf = torch.zeros(num_envs, episode_len, actor.nu, device=device, dtype=actor.dtype)
        rewards_buf = torch.zeros(num_envs, episode_len, device=device, dtype=actor.dtype)
        log_probs_buf = torch.zeros(num_envs, episode_len, device=device, dtype=actor.dtype)
        current_states = envs.reset()
        with torch.no_grad():
            for t in range(episode_len):
                actions, log_probs, _, _ = actor(current_states, deterministic=False)
                next_states, rewards, dones, truncated, _ = envs.step(actions)
                states_buf[:, t], actions_buf[:, t], rewards_buf[:, t], log_probs_buf[:, t] = current_states, actions, rewards, log_probs
                for i in range(num_envs):
                    if dones[i] or truncated[i]:
                        reset_obs, _ = envs.envs[i].reset()
                        next_states[i] = torch.from_numpy(reset_obs).to(device, dtype=actor.dtype)
                current_states = next_states
        logging.info(f" Raccolta dati completata.")
        logging.info(f"Fase 1.5: Creazione buffer di sequenze storiche (history_len={history_len})...")
        b_history_states = create_history_buffers(states_buf, history_len).reshape(-1, history_len, obs_dim)
        b_history_actions = create_history_buffers(actions_buf, history_len).reshape(-1, history_len, actor.nu)
        b_timesteps = torch.arange(history_len, device=device, dtype=torch.long).repeat(b_history_states.shape[0], 1)

        with torch.no_grad():
            if isinstance(critic, CriticDecisionTransformer):
                temp_values = critic(b_history_states, b_history_actions, b_timesteps)
                values_buf = temp_values[:, -1].reshape(num_envs, episode_len)
            else:
                values_buf = critic(states_buf.view(-1, 1, obs_dim)).reshape(num_envs, episode_len)

        logging.info("Fase 2: Calcolo Valori e Vantaggi (GAE)...")
        returns, advantages = batched_compute_gae(rewards_buf, values_buf)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        logging.info(" Calcolo GAE completato.")

        logging.info(f"Fase 3: Aggiornamento PPO ({ppo_epochs} epoche)...")
        # ... (Fase 3 - Aggiornamento PPO, rimane invariata)
        b_states = states_buf.view(-1, obs_dim)
        b_actions = actions_buf.view(-1, actor.nu)
        b_log_probs_old = log_probs_buf.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        ppo_batch_size = b_states.shape[0]
        mini_batch_size = min(3072, ppo_batch_size)
        for epoch in range(ppo_epochs):
            permutation = torch.randperm(ppo_batch_size)
            for start in range(0, ppo_batch_size, mini_batch_size):
                end = start + mini_batch_size
                indices = permutation[start:end]
                mb_states = b_states[indices]
                mb_actions = b_actions[indices]
                mb_log_probs_old = b_log_probs_old[indices]
                mb_advantages = b_advantages[indices]
                mb_returns = b_returns[indices]
                mb_history_states = b_history_states[indices]
                mb_history_actions = b_history_actions[indices]
                mb_timesteps = b_timesteps[indices]
                with autocast(device_type=str(device).split(":")[0], dtype=torch.float16, enabled=use_amp):
                    new_log_probs, entropy = actor.evaluate_actions(mb_states, mb_actions)
                    ratio = torch.exp(new_log_probs - mb_log_probs_old)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -entropy.mean()
                    if isinstance(critic, CriticDecisionTransformer):
                        all_values = critic(mb_history_states, mb_history_actions, mb_timesteps)
                        new_values = all_values[:, -1]
                    else:
                        new_values = critic(mb_states.unsqueeze(1)).squeeze(1)
                    critic_loss = F.mse_loss(new_values, mb_returns)
                    total_loss = actor_loss + entropy_coeff * entropy_loss + critic_loss
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.step(actor_optimizer)
                scaler.step(critic_optimizer)
                scaler.update()

        logging.info(f" Aggiornamento PPO completato.")
        mean_reward = rewards_buf.mean().item()
        training_rewards.append(mean_reward)
        logging.info(
            f"📊 Risultati Step: Reward Medio: {mean_reward:.3f} | Loss Attore: {actor_loss.item():.4f} | Loss Critico: {critic_loss.item():.4f}"
        )

        if checkpoint_manager:
            ### <<< MODIFICA CHIAVE: Reset dello stato interno dell'attore prima di salvare >>> ###
            with torch.no_grad():
                # Creiamo un input fittizio con batch_size = 1
                dummy_input_dim = actor.cost_map_net[0].in_features
                dummy_input = torch.zeros(1, dummy_input_dim, device=device, dtype=actor.dtype)
                # Questa chiamata resetta i buffer interni (x_ref, u_ref) a batch_size = 1
                actor._update_cost_module(dummy_input)

            checkpoint_manager.save(
                completed_step_idx=step_idx,
                actor=actor, critic=critic,
                actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                scaler=scaler, current_reward=mean_reward
            )

        logging.info(f"--- Step {step_idx + 1} completato in {time.time() - step_start_time:.2f}s ---\n")

    envs.close()
    return training_rewards