import torch
import numpy as np
from typing import Callable, List, Tuple, Any

class ParallelEnvManager:
    """
    Un gestore per eseguire istanze multiple di un ambiente in parallelo.
    """

    def __init__(self, env_fn: Callable, num_envs: int, device: torch.device):
        """
        Inizializza N ambienti.
        """
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.device = device

        if hasattr(self.envs[0], 'observation_space') and hasattr(self.envs[0], 'action_space'):
            self.single_observation_space_shape = self.envs[0].observation_space.shape
            self.single_action_space_shape = self.envs[0].action_space.shape
        else:
            print("Attenzione: L'ambiente non sembra avere gli attributi 'observation_space' o 'action_space'.")

    def reset(self) -> torch.Tensor:
        """
        Resetta tutti gli ambienti e restituisce un tensore delle osservazioni.
        """
        # FIX: gym.reset() restituisce una tupla (osservazione, info).
        # Estraiamo solo il primo elemento (l'osservazione).
        observations = [env.reset()[0] for env in self.envs]
        return torch.from_numpy(np.stack(observations)).to(self.device, dtype=torch.float32)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        """
        Esegue un passo in tutti gli ambienti con le azioni fornite.
        """
        actions_np = actions.cpu().numpy()
        next_states, rewards, terminateds, truncateds, infos = [], [], [], [], []

        for i, env in enumerate(self.envs):
            ns, r, terminated, truncated, info = env.step(actions_np[i])
            next_states.append(ns)
            rewards.append(r)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        return (
            torch.from_numpy(np.stack(next_states)).to(self.device, dtype=torch.float32),
            torch.from_numpy(np.array(rewards)).to(self.device, dtype=torch.float32),
            torch.from_numpy(np.array(terminateds)).to(self.device, dtype=torch.bool),
            torch.from_numpy(np.array(truncateds)).to(self.device, dtype=torch.bool),
            infos
        )

    def close(self):
        """Chiude tutti gli ambienti."""
        [env.close() for env in self.envs]