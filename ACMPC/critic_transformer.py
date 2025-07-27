# File: critic_decision_transformer.py

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Optional


class CriticTransformer(nn.Module):
    """
    Un critico basato su Transformer che stima il valore V(s) a partire da uno stato.
    Progettato per ricevere sequenze di token di stato.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,  # Mantenuto per coerenza di firma ma non usato
            history_len: int,
            pred_horizon: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.token_dim = state_dim

        # Layer per proiettare i token di stato nello spazio nascosto del Transformer
        self.embed = nn.Linear(self.token_dim, hidden_size)

        # La sequenza più lunga che il critico può vedere è la traiettoria predetta dall'MPC
        max_seq_length = pred_horizon + history_len

        # Configurazione standard di un Transformer (stile BERT)
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * hidden_size,
            vocab_size=1,  # Non usato quando si forniscono embedding direttamente
            max_position_embeddings=max_seq_length,
            is_decoder=True,  # Gestisce la maschera causale internamente
        )
        self.transformer = BertModel(config)

        # Testa di regressione per mappare l'output del Transformer a un valore scalare
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        """
        Restituisce un valore V(s) per OGNI token nella sequenza di input.
        L'input `sequence_tokens` dovrebbe essere una sequenza di stati.
        Shape: (batch_size, seq_len, state_dim)
        """
        # 1. Proietta i token di input nello spazio di embedding
        x = self.embed(sequence_tokens)

        # 2. Passa gli embedding attraverso il Transformer
        outputs = self.transformer(inputs_embeds=x)

        # 3. Prendi l'output di tutti i token dall'ultimo layer
        all_token_outputs = outputs.last_hidden_state

        # 4. Applica la testa di regressione per ottenere i valori
        values = self.head(all_token_outputs)

        # Rimuovi l'ultima dimensione (che è 1)
        return values.squeeze(-1)


class CriticDecisionTransformer(nn.Module):
    """
    Un critico ispirato ai Decision Transformer che accetta sequenze di stati e azioni.
    Questo gli permette di essere adattivo al contesto della traiettoria per compensare
    le incertezze del modello dinamico.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            history_len: int,
            pred_horizon: int,
            hidden_size: int = 256,
            num_layers: int = 4,
            num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Layer di embedding separati per stati e azioni
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(action_dim, hidden_size)

        # L'embedding del timestep fornisce informazioni posizionali/temporali
        # alla sequenza, un elemento chiave nei Transformer.
        max_len = pred_horizon + history_len
        self.embed_timestep = nn.Embedding(max_len, hidden_size)

        # LayerNorm per stabilizzare gli embedding prima di passarli al Transformer
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Configurazione del Transformer
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * hidden_size,
            vocab_size=1,
            max_position_embeddings=max_len,
            is_decoder=True,
        )
        self.transformer = BertModel(config)

        # Testa di regressione per mappare l'output a un valore scalare
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Elabora sequenze di stati e azioni per produrre una sequenza di valori.
        - states: (batch_size, seq_len, state_dim)
        - actions: (batch_size, seq_len, action_dim)
        - timesteps: (batch_size, seq_len)
        """
        # 1. Calcola gli embedding per stati, azioni e timestep
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        # 2. Somma gli embedding per creare i token finali.
        #    Questa somma permette di combinare le informazioni da diverse modalità
        #    (stato, azione, tempo) in un'unica rappresentazione.
        token_embeddings = state_embeddings + action_embeddings + time_embeddings
        token_embeddings = self.embed_ln(token_embeddings)

        # 3. Passa i token al Transformer
        transformer_outputs = self.transformer(inputs_embeds=token_embeddings)
        hidden_states = transformer_outputs.last_hidden_state

        # 4. Calcola i valori per ogni token nella sequenza
        values = self.head(hidden_states)

        return values.squeeze(-1)