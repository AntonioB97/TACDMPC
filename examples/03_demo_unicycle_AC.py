import torch
import numpy as np
import gym
from gym import spaces
import os
import math
import matplotlib.pyplot as plt
from typing import Dict

# Assumiamo che i file del framework siano in una cartella/modulo 'ACMPC'
from ACMPC import ActorMPC, CheckpointManager, CriticTransformer, train


# =============================================================================
# 1. DEFINIZIONE DELL'AMBIENTE GYM CON LE MODIFICHE RICHIESTE
# =============================================================================

class UnicycleEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, trajectory: np.ndarray, dt: float = 0.1, episode_len: int = 250):
        super().__init__()
        self.dt, self.trajectory, self.max_steps = dt, trajectory, episode_len
        self.state_dim, self.state = 3, np.zeros(3, dtype=np.float32)  # [x, y, theta]

        # <-- MODIFICA: Aumentati i limiti di velocità per un comportamento più aggressivo
        self.action_space = spaces.Box(
            low=np.array([0.0, -4.0]),  # Velocità lineare (v), Velocità angolare (omega)
            high=np.array([8.0, 4.0]),
            dtype=np.float32
        )

        # <-- MODIFICA: L'osservazione ora contiene solo lo stato e il waypoint CORRENTE
        self.observation_dim = self.state_dim + 2  # 3 (stato) + 2 (pos waypoint) = 5
        obs_high = np.full(self.observation_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.current_step, self.waypoint_index = 0, 0
        self.goal_radius, self.fail_radius = 0.5, 7.0  # Aumentato fail_radius per dare più spazio
        self.previous_dist_to_wp = 0.0

    def _get_obs(self):
        # <-- MODIFICA: Fornisce solo il waypoint corrente.
        # Se tutti i waypoint sono stati raggiunti, continua a puntare all'ultimo.
        current_wp_idx = min(self.waypoint_index, len(self.trajectory) - 1)
        current_wp = self.trajectory[current_wp_idx]
        return np.concatenate([self.state, current_wp]).astype(np.float32)

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.current_step, self.waypoint_index = 0, 0
        current_wp = self.trajectory[self.waypoint_index]
        self.previous_dist_to_wp = np.linalg.norm(self.state[:2] - current_wp)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        v, omega = action[0], action[1]
        x, y, theta = self.state
        x_new, y_new = x + v * math.cos(theta) * self.dt, y + v * math.sin(theta) * self.dt
        theta_new = (theta + omega * self.dt + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([x_new, y_new, theta_new], dtype=np.float32)
        self.current_step += 1
        terminated, truncated = False, self.current_step >= self.max_steps

        # --- Logica della Reward Modificata ---
        current_wp_idx = min(self.waypoint_index, len(self.trajectory) - 1)
        current_wp = self.trajectory[current_wp_idx]
        dist_to_wp = np.linalg.norm(self.state[:2] - current_wp)

        # 1. Reward di progresso (peso alto per renderla prioritaria)
        reward_progress = 5.0 * (self.previous_dist_to_wp - dist_to_wp)
        self.previous_dist_to_wp = dist_to_wp

        # 2. <-- MODIFICA: Reward per il movimento più forte per incentivare l'azione
        reward_movement = 0.2 * v

        # 3. Reward di orientamento (invariata)
        angle_to_wp = math.atan2(current_wp[1] - y, current_wp[0] - x)
        reward_orientation = 0.1 * (math.cos(abs(theta - angle_to_wp)))

        # 4. <-- MODIFICA: Penalità per le azioni molto ridotta
        action_penalty = 0.0001 * np.sum(np.square(action))
        time_penalty = 0.01

        reward = reward_progress + reward_movement + reward_orientation - action_penalty - time_penalty

        # 5. Bonus/Malus per eventi sparsi
        # Controlla solo se non abbiamo già finito la traiettoria
        if self.waypoint_index < len(self.trajectory) and dist_to_wp < self.goal_radius:
            reward += 20.0
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.trajectory):
                terminated = True
                reward += 100.0
            else:
                next_wp = self.trajectory[self.waypoint_index]
                self.previous_dist_to_wp = np.linalg.norm(self.state[:2] - next_wp)

        if dist_to_wp > self.fail_radius:
            reward -= 20.0
            terminated = True

        info = {'dist_to_wp': dist_to_wp, 'vel_linear': v, 'vel_angular': omega}
        return self._get_obs(), float(reward), terminated, truncated, info


gym.register(id='UnicyclePathV2-v0', entry_point=__name__ + ':UnicycleEnv')


# =============================================================================
# 2. FUNZIONI PER L'MPC E RICOMPENSA
# =============================================================================
def f_dyn_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    v, omega, theta = u[..., 0], u[..., 1], x[..., 2]
    new_x = x[..., 0] + v * torch.cos(theta) * dt
    new_y = x[..., 1] + v * torch.sin(theta) * dt
    new_theta = (theta + omega * dt + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.stack([new_x, new_y, new_theta], dim=-1)


def f_dyn_jac_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    B, dtype, device = x.shape[0], x.dtype, x.device
    v, theta = u[..., 0], x[..., 2]
    A = torch.zeros(B, 3, 3, dtype=dtype, device=device)
    A[:, [0, 1, 2], [0, 1, 2]] = 1.0
    A[:, 0, 2], A[:, 1, 2] = -v * torch.sin(theta) * dt, v * torch.cos(theta) * dt
    B_mat = torch.zeros(B, 3, 2, dtype=dtype, device=device)
    B_mat[:, 0, 0], B_mat[:, 1, 0] = torch.cos(theta) * dt, torch.sin(theta) * dt
    B_mat[:, 2, 1] = dt
    return A, B_mat


def unicycle_mpve_reward_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # <-- MODIFICA: Adattato alla nuova dimensione di osservazione
    pos, target_wp = state[..., :2], state[..., 3:5]
    dist_to_wp = torch.linalg.norm(pos - target_wp, dim=-1)
    return -dist_to_wp


# =============================================================================
# 3. FUNZIONI DI VALUTAZIONE E PLOTTING (INVARIATE)
# =============================================================================
def run_evaluation(actor: ActorMPC, env_fn, num_episodes: int, final_target: np.ndarray, device: str) -> Dict:
    actor.eval()
    all_Xs, all_Us = [], []
    print(f"\nEsecuzione valutazione per {num_episodes} episodi...")
    for i in range(num_episodes):
        env, (done, truncated) = env_fn(), (False, False)
        obs, _ = env.reset()
        episode_states, episode_actions = [obs[:3]], []
        while not (done or truncated):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = actor(obs_tensor, deterministic=True)
            obs, _, done, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            episode_states.append(obs[:3])
            episode_actions.append(action.squeeze(0).cpu().numpy())
        all_Xs.append(np.array(episode_states));
        all_Us.append(np.array(episode_actions))
        env.close()
    max_len = max(len(x) for x in all_Xs)
    padded_Xs = torch.from_numpy(np.array([np.pad(x, ((0, max_len - len(x)), (0, 0)), 'edge') for x in all_Xs])).float()
    padded_Us = torch.from_numpy(
        np.array([np.pad(u, ((0, max_len - 1 - len(u)), (0, 0)), 'edge') for u in all_Us])).float()
    final_states = np.array([ep[-1, :] for ep in all_Xs])
    errors_combined = np.hstack([np.abs(final_states[:, :2] - final_target), np.abs(final_states[:, 2])[:, None]])
    return {'Xs': padded_Xs, 'Us': padded_Us, 'dt': env.dt,
            'metrics': {'mean_final_errors': np.mean(errors_combined, axis=0),
                        'var_final_errors': np.var(errors_combined, axis=0)}}


def _plot_unicycle_results(results: Dict, trajectory: np.ndarray, method_name: str = 'AC-MPC'):
    fig, axs = plt.subplots(2, 2, figsize=(22, 16));
    fig.suptitle(f"Analisi Statistica: {method_name} (Unicycle)", fontsize=22, weight='bold')
    dt = results['dt'];
    mean_xs, std_xs = results['Xs'].mean(dim=0).cpu(), results['Xs'].std(dim=0).cpu();
    mean_us, std_us = results['Us'].mean(dim=0).cpu(), results['Us'].std(dim=0).cpu();
    time_ax_xs, time_ax_us = torch.arange(mean_xs.shape[0]) * dt, torch.arange(mean_us.shape[0]) * dt
    ax = axs[0, 0];
    ax.plot(mean_xs[:, 0], mean_xs[:, 1], 'b-', label='Traiettoria Media');
    ax.fill_between(mean_xs[:, 0], mean_xs[:, 1] - std_xs[:, 1], mean_xs[:, 1] + std_xs[:, 1], color='b', alpha=0.2);
    ax.fill_betweenx(mean_xs[:, 1], mean_xs[:, 0] - std_xs[:, 0], mean_xs[:, 0] + std_xs[:, 0], color='b', alpha=0.2);
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro--', label='Waypoints Target');
    ax.plot(0, 0, 'ko', markersize=8, label='Start');
    ax.set_title("Traiettoria Media nel Piano XY");
    ax.set_xlabel("X [m]");
    ax.set_ylabel("Y [m]");
    ax.grid(True);
    ax.legend();
    ax.axis('equal')
    ax = axs[0, 1];
    ax.plot(time_ax_xs, mean_xs[:, 0], label='Media (x)');
    ax.fill_between(time_ax_xs, mean_xs[:, 0] - std_xs[:, 0], mean_xs[:, 0] + std_xs[:, 0], alpha=0.2);
    ax.plot(time_ax_xs, mean_xs[:, 1], label='Media (y)');
    ax.fill_between(time_ax_xs, mean_xs[:, 1] - std_xs[:, 1], mean_xs[:, 1] + std_xs[:, 1], alpha=0.2);
    ax.plot(time_ax_xs, mean_xs[:, 2], label=r'Media ($\theta$)');
    ax.fill_between(time_ax_xs, mean_xs[:, 2] - std_xs[:, 2], mean_xs[:, 2] + std_xs[:, 2], alpha=0.2);
    ax.set_title("Evoluzione Stati Fisici");
    ax.set_xlabel("Tempo [s]");
    ax.set_ylabel("Valore [m, rad]");
    ax.grid(True);
    ax.legend()
    ax = axs[1, 0];
    ax.plot(time_ax_us, mean_us[:, 0], label='Media Vel. Lineare (v)');
    ax.fill_between(time_ax_us, mean_us[:, 0] - std_us[:, 0], mean_us[:, 0] + std_us[:, 0], alpha=0.2);
    ax.plot(time_ax_us, mean_us[:, 1], label='Media Vel. Angolare ($\omega$)');
    ax.fill_between(time_ax_us, mean_us[:, 1] - std_us[:, 1], mean_us[:, 1] + std_us[:, 1], alpha=0.2);
    ax.set_title("Input di Controllo");
    ax.set_xlabel("Tempo [s]");
    ax.set_ylabel("Valore [m/s, rad/s]");
    ax.grid(True);
    ax.legend()
    ax = axs[1, 1];
    metrics = results['metrics'];
    bar_labels = ['Errore X [m]', 'Errore Y [m]', r'Errore $\theta$ [rad]'];
    ax.bar(bar_labels, metrics['mean_final_errors'], yerr=np.sqrt(metrics['var_final_errors']), capsize=5,
           color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8);
    ax.set_title("Errore Assoluto Medio Finale");
    ax.set_ylabel("Errore Assoluto Medio");
    ax.grid(True, axis='y')
    fig.tight_layout(rect=[0, 0, 1, 0.96]);
    plt.savefig("evaluation_results_unicycle.png");
    plt.show()


def _plot_single_trajectory(actor, env_fn, device, trajectory):
    print("\nVisualizzazione di una singola traiettoria di esempio...");
    env = env_fn();
    obs, _ = env.reset();
    history_x, history_y = [], [];
    done, truncated = False, False
    while not (done or truncated):
        history_x.append(env.state[0]);
        history_y.append(env.state[1])
        with torch.no_grad(): action, _, _, _ = actor(torch.from_numpy(obs).to(device).float(), deterministic=True)
        obs, _, done, truncated, _ = env.step(action.cpu().numpy().squeeze(0))
    plt.figure(figsize=(12, 8));
    target_path = np.vstack([[0, 0], trajectory]);
    plt.plot(target_path[:, 0], target_path[:, 1], 'go--', label='Traiettoria Target', linewidth=2, markersize=10);
    plt.plot(history_x, history_y, 'b-', label='Traiettoria Eseguita', linewidth=2);
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro', markersize=10, label='Waypoints');
    plt.plot(0, 0, 'ko', markersize=10, label='Start');
    plt.title("Esempio di Traiettoria Eseguita con AC-MPC");
    plt.xlabel("X (m)");
    plt.ylabel("Y (m)");
    plt.legend();
    plt.grid(True);
    plt.axis('equal');
    plt.savefig("trajectory_unicycle_single_run.png");
    plt.show()


# =============================================================================
# 4. SCRIPT DI TRAINING PRINCIPALE
# =============================================================================
if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED, CHECKPOINT_DIR = 42, "./checkpoints_unicycle_aggressive"

    # <-- MODIFICA: Parametri di training ripristinati per coerenza
    DT, NX, NU = 0.1, 3, 2
    OBS_DIM = 5  # <-- MODIFICA: Aggiornato alla nuova dimensione
    MPC_HORIZON, EPISODE_LEN = 15, 250
    TOTAL_TRAINING_STEPS, NUM_ENVS, PPO_EPOCHS = 200, 24, 4

    print(f"Utilizzando il dispositivo: {DEVICE}");
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    trajectory = np.array([[2.0, 2.0], [4.0, 1.0]], dtype=np.float32)
    print(f"Traiettoria target: Start(0,0) -> WP1{tuple(trajectory[0])} -> WP2{tuple(trajectory[1])}")

    env_fn = lambda: gym.make('UnicyclePathV2-v0', trajectory=trajectory, dt=DT, episode_len=EPISODE_LEN)
    bound_f_dyn = lambda x, u, _: f_dyn_torch(x, u, dt=DT)
    bound_f_dyn_jac = lambda x, u, _: f_dyn_jac_torch(x, u, dt=DT)

    # <-- MODIFICA: Inizializzazione di Attore e Critico con la nuova OBS_DIM
    actor = ActorMPC(
        nx=NX, nu=NU, observation_dim=OBS_DIM, horizon=MPC_HORIZON, dt=DT,
        f_dyn=bound_f_dyn, f_dyn_jac=bound_f_dyn_jac, device=DEVICE, grad_method='analytic'
    ).to(DEVICE)

    critic = CriticTransformer(
        state_dim=OBS_DIM, action_dim=NU, history_len=1, pred_horizon=MPC_HORIZON,
        hidden_size=256, num_layers=4, num_heads=4
    ).to(DEVICE)

    checkpoint_manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)
    print("\n>>> Inizio del training per AC-MPC su Unicycle (versione aggressiva) <<<")

    train(
        env_fn=env_fn, actor=actor, critic=critic, reward_fn=unicycle_mpve_reward_fn,
        steps=TOTAL_TRAINING_STEPS, num_envs=NUM_ENVS, episode_len=EPISODE_LEN,
        mpc_horizon=MPC_HORIZON, ppo_epochs=PPO_EPOCHS, clip_param=0.2,
        entropy_coeff=0.01, mpve_coeff=0.1, checkpoint_manager=checkpoint_manager,
        resume_from="best", use_amp=(DEVICE == "cuda"), seed=SEED
    )

    print("\n>>> Training completato! <<<")
    checkpoint_manager.load(device=DEVICE, resume_from="best", actor=actor, critic=critic)
    eval_results = run_evaluation(
        actor=actor, env_fn=env_fn, num_episodes=50,
        final_target=trajectory[-1], device=DEVICE
    )

    metrics = eval_results['metrics']
    state_labels = ["Posizione X", "Posizione Y", "Angolo Theta"]
    print("\n>>> Metriche di Valutazione (calcolate sugli stati finali di 50 episodi) <<<")
    print("-" * 70);
    print("Errore Assoluto Medio Finale per Stato:")
    for i, label in enumerate(state_labels): print(f"  - {label:<20}: {metrics['mean_final_errors'][i]:.4f}")
    print("-" * 70);
    print("Varianza dell'Errore Assoluto Finale per Stato:")
    for i, label in enumerate(state_labels): print(f"  - {label:<20}: {metrics['var_final_errors'][i]:.4f}")
    print("-" * 70)

    _plot_unicycle_results(eval_results, trajectory, method_name='AC-MPC (Aggressive)')
    _plot_single_trajectory(actor, env_fn, DEVICE, trajectory)
