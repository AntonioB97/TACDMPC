import torch
import numpy as np
import gym
from gym import spaces
import os
import matplotlib.pyplot as plt
import time

# Assumiamo che i file del framework siano in una cartella/modulo 'ACMPC'
from ACMPC import ActorMPC
# ### MODIFICA: Importiamo entrambi i critici per flessibilità ###
from ACMPC import CriticTransformer, CriticDecisionTransformer
from ACMPC import train
from ACMPC import CheckpointManager


# =============================================================================
# 1. DEFINIZIONE DELL'AMBIENTE DINAMICO
# =============================================================================

class DynamicWaypointEnv(gym.Env):
    """
    Ambiente con waypoint dinamici e una funzione di reward basata sul progresso
    per un apprendimento più stabile e denso.
    """

    def __init__(self, dt: float, episode_len: int, num_target_waypoints: int = 4, max_dist: float = 10.0):
        super().__init__()
        self.dt = dt
        self.max_steps = episode_len
        self.num_target_waypoints = num_target_waypoints
        self.max_dist_from_prev = max_dist

        self.state = np.zeros(2, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-10.0]), high=np.array([10.0]), dtype=np.float32)

        self.observation_dim = 3
        obs_high = np.full(self.observation_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.current_step = 0
        self.goal_radius = 0.5
        self.target_waypoint = np.zeros(1, dtype=np.float32)
        self.waypoints_reached = 0
        self.prev_dist_to_wp = 0.0

    def _get_obs(self):
        return np.concatenate([self.state, self.target_waypoint]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)
        self.current_step = 0
        self.waypoints_reached = 0
        self.target_waypoint = self.np_random.uniform(low=-5.0, high=5.0, size=(1,)).astype(np.float32)
        self.prev_dist_to_wp = np.linalg.norm(self.state[0] - self.target_waypoint[0])
        return self._get_obs(), {}

    def step(self, action):
        pos, vel = self.state
        force = np.clip(action, self.action_space.low, self.action_space.high)[0]
        new_vel = vel + force * self.dt
        new_pos = pos + new_vel * self.dt
        self.state = np.array([new_pos, new_vel], dtype=np.float32)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        dist_to_wp = np.linalg.norm(self.state[0] - self.target_waypoint[0])

        progress = self.prev_dist_to_wp - dist_to_wp
        action_penalty = 0.01 * (force ** 2)
        reward = 10.0 * progress - action_penalty
        self.prev_dist_to_wp = dist_to_wp

        terminated = False
        if dist_to_wp < self.goal_radius:
            reward += 10.0
            self.waypoints_reached += 1
            if self.waypoints_reached >= self.num_target_waypoints:
                terminated = True
            else:
                current_wp_pos = self.target_waypoint[0]
                offset = self.np_random.uniform(low=-self.max_dist_from_prev, high=self.max_dist_from_prev)
                new_wp_pos = current_wp_pos + offset
                self.target_waypoint = np.array([new_wp_pos], dtype=np.float32)
                self.prev_dist_to_wp = np.linalg.norm(self.state[0] - self.target_waypoint[0])

        return self._get_obs(), float(reward), terminated, truncated, {}


gym.register(id='DynamicWaypoint-v0', entry_point=__name__ + ':DynamicWaypointEnv')


# =============================================================================
# 2. FUNZIONI PER L'MPC
# =============================================================================
def f_dyn_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    pos, vel = x[..., 0:1], x[..., 1:2]
    new_vel = vel + u * dt
    new_pos = pos + new_vel * dt
    return torch.cat([new_pos, new_vel], dim=-1)


def f_dyn_jac_torch(x: torch.Tensor, u: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.shape[0]
    device, dtype = x.device, x.dtype
    A = torch.tensor([[1, dt], [0, 1]], device=device, dtype=dtype).expand(batch_size, -1, -1)
    B = torch.tensor([[dt * dt / 2], [dt]], device=device, dtype=dtype).expand(batch_size, -1, -1)
    return A, B


# =============================================================================
# 3. FUNZIONE DI VALUTAZIONE
# =============================================================================
def run_and_plot_evaluation_dynamic(actor: ActorMPC, env_fn, device: str, save_path: str):
    """
    Esegue una valutazione dell'agente in un ambiente con waypoint dinamici
    e produce due grafici:
    1. La traiettoria della posizione dell'agente rispetto ai waypoint.
    2. La cronologia degli input (azioni) applicati.
    """
    print("\nEsecuzione valutazione finale con waypoint dinamici...")
    actor.eval()
    env = env_fn()
    obs, _ = env.reset()

    # Cronologia per il plotting
    history_pos = [obs[0]]
    history_target_wp = [obs[2]]
    history_actions = []  # <<< AGGIUNTA

    done, truncated = False, False
    while not (done or truncated):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_tensor, _, _, _ = actor(obs_tensor, deterministic=True)
        action = action_tensor.squeeze(0).cpu().numpy()

        # Salviamo l'azione prima dello step
        history_actions.append(action[0])  # <<< AGGIUNTA

        obs, _, done, truncated, _ = env.step(action)

        # Salviamo lo stato risultante
        history_pos.append(obs[0])
        history_target_wp.append(obs[2])

    env.close()

    # --- INIZIO BLOCCO MODIFICATO ---

    # Creazione degli assi temporali
    # 'time_axis' per gli stati (es. posizione)
    time_axis = np.arange(len(history_pos)) * env.dt
    # 'action_time_axis' per le azioni, che sono una in meno
    action_time_axis = time_axis[:-1]

    # Creazione della figura con due subplot verticali che condividono l'asse x
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Valutazione AC-MPC con Critico Adattivo (Decision Transformer)", fontsize=16)

    # --- Subplot 1: Posizione vs Tempo ---
    axes[0].plot(time_axis, history_pos, label='Posizione Agente', color='b', zorder=5)
    axes[0].plot(time_axis, history_target_wp, label='Waypoint Target Dinamico', color='r', linestyle='--',
                 drawstyle='steps-post', zorder=4)
    axes[0].set_ylabel("Posizione")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title("Traiettoria dell'Agente")

    # --- Subplot 2: Ingressi (Azioni) vs Tempo ---
    axes[1].plot(action_time_axis, history_actions, label='Ingresso (Forza)', color='purple', drawstyle='steps-post')
    axes[1].set_ylabel("Forza Applicata")
    axes[1].set_xlabel("Tempo (s)")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title("Cronologia degli Ingressi")

    # Trova e segna i punti di cambio del waypoint su entrambi i grafici
    change_indices = np.where(np.abs(np.diff(np.array(history_target_wp))) > 1e-5)[0]

    for i, point_idx in enumerate(change_indices):
        change_time = time_axis[point_idx + 1]

        # Aggiungi linea verticale e testo al primo subplot (posizione)
        axes[0].axvline(x=change_time, color='g', linestyle=':', alpha=0.7)
        axes[0].text(change_time + 0.5, history_target_wp[point_idx + 1], f'Nuovo WP {i + 1}', color='g', ha='left',
                     va='center')

        # Aggiungi solo la linea verticale al secondo subplot (ingressi)
        axes[1].axvline(x=change_time, color='g', linestyle=':', alpha=0.7)

    # Ottimizza il layout e salva la figura
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Aggiusta lo spazio per il suptitle
    plt.savefig(save_path)
    print(f"Grafico di valutazione salvato in '{save_path}'")
    plt.show()

    # --- FINE BLOCCO MODIFICATO ---


# =============================================================================
# 4. SCRIPT DI TRAINING PRINCIPALE
# =============================================================================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    CHECKPOINT_DIR = "checkpoints_decision_transformer_v1"

    # Parametri di training
    DT = 0.2
    MPC_HORIZON = 3
    NX, NU = 2, 1
    EPISODE_LEN = 200
    TOTAL_TRAINING_STEPS = 60
    NUM_ENVS, PPO_EPOCHS = 64, 2
    HISTORY_LEN = 8

    print(f"Utilizzando il dispositivo: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    OBS_DIM = 3
    print(f"Ambiente dinamico | Dimensione Osservazione: {OBS_DIM}")
    env_fn = lambda: gym.make('DynamicWaypoint-v0', dt=DT, episode_len=EPISODE_LEN)
    # Convert the control limits to torch tensors
    u_min_tensor = torch.tensor([-10.0], dtype=torch.float32, device=DEVICE)
    u_max_tensor = torch.tensor([10.0], dtype=torch.float32, device=DEVICE)

    actor = ActorMPC(
        nx=NX, nu=NU, observation_dim=OBS_DIM, horizon=MPC_HORIZON, dt=DT,
        f_dyn=f_dyn_torch, f_dyn_jac=f_dyn_jac_torch, device=DEVICE, grad_method='analytic',
        u_min=u_min_tensor,  # Pass the tensor
        u_max=u_max_tensor  # Pass the tensor
    ).to(DEVICE)

    ### <<< MODIFICA: Istanziamo il nuovo CriticDecisionTransformer >>> ###
    critic = CriticDecisionTransformer(
        state_dim=OBS_DIM,
        action_dim=NU,
        history_len=HISTORY_LEN,
        pred_horizon=MPC_HORIZON,
        hidden_size=256,
        num_layers=4,
        num_heads=8
    ).to(DEVICE)

    checkpoint_manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)

    print("\n>>> Inizio del training con Critico Adattivo (Decision Transformer) <<<")
    ### <<< MODIFICA: Passiamo history_len alla funzione di training >>> ###
    train(
        env_fn=env_fn,
        actor=actor,
        critic=critic,
        reward_fn=None,
        history_len=HISTORY_LEN,
        steps=TOTAL_TRAINING_STEPS,
        num_envs=NUM_ENVS,
        episode_len=EPISODE_LEN,
        mpc_horizon=MPC_HORIZON,
        ppo_epochs=PPO_EPOCHS,
        clip_param=0.2,
        entropy_coeff=0.005,
        mpve_coeff=0.0,  # MPVE è disattivato per semplicità con questo critico
        checkpoint_manager=checkpoint_manager,
        resume_from="latest",
        use_amp=(DEVICE == "cuda"),
        seed=SEED
    )
    print("\n>>> Training completato! <<<")

    checkpoint_manager.load(device=DEVICE, resume_from="best", actor=actor, critic=critic)
    run_and_plot_evaluation_dynamic(
        actor, env_fn, DEVICE,
        save_path="evaluation_decision_transformer_v1.png"
    )


if __name__ == '__main__':
    main()