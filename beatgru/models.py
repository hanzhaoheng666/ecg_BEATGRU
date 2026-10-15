"""Current-beat CNN and record-state causal GRU."""
from __future__ import annotations

import torch
from torch import nn


class BeatCNN(nn.Module):
    """Morphology/RR classifier used as the frozen emission model."""
    def __init__(self, channels: int = 3, rr_dim: int = 8, head_hidden: int = 96,
                 n_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, 9, padding=4), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv1d(96, 128, 5, padding=2), nn.GroupNorm(8, 128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rr = nn.Sequential(nn.Linear(rr_dim, 32), nn.LayerNorm(32), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(160), nn.Linear(160, head_hidden), nn.GELU(), nn.Dropout(0.15), nn.Linear(head_hidden, n_classes)
        )

    def encode(self, waveform: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        """Current-beat representation; it contains no across-beat context."""
        return torch.cat((self.conv(waveform).flatten(1), self.rr(rr)), dim=1)

    def forward(self, waveform: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(waveform, rr))


class MorphologyCNN(nn.Module):
    """M0 morphology-only classifier: raw+d1+d2, with no RR branch."""
    def __init__(self, channels: int = 3, head_hidden: int = 96) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, 9, padding=4), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv1d(96, 128, 5, padding=2), nn.GroupNorm(8, 128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, head_hidden), nn.GELU(), nn.Dropout(.15), nn.Linear(head_hidden, 3))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(waveform).flatten(1))


class LogitGRU(nn.Module):
    """A causal residual correction of frozen CNN logits.

    The state is supplied by the caller.  It must be reset at every record
    boundary; no labels are an input to this module.
    """
    def __init__(self, rr_dim: int = 8, hidden: int = 128, max_correction: float = 0.75,
                 n_classes: int = 3) -> None:
        super().__init__()
        self.max_correction = float(max_correction)
        self.input = nn.Sequential(nn.Linear(n_classes + rr_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.delta = nn.Linear(hidden, n_classes)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward_chunk(self, base_logits: torch.Tensor, rr: torch.Tensor,
                      state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.input(torch.cat((base_logits, rr), dim=1))
        history, new_state = self.gru(token.unsqueeze(0), state)
        return base_logits + self.max_correction * self.delta(history.squeeze(0)), new_state


class ShortLogitGRU(nn.Module):
    """Strict sliding-window history decoder.

    Each output is computed from exactly the current and previous at most
    ``window`` tokens.  Windows are batched with their true lengths, so no
    hidden state is carried across the 32-beat boundary and no future token is
    visible.  This is deliberately different from periodically resetting a
    recurrent state.
    """
    def __init__(self, rr_dim: int = 8, hidden: int = 128, max_correction: float = .75,
                 window: int = 32) -> None:
        super().__init__()
        self.window = int(window); self.max_correction = float(max_correction)
        self.input = nn.Sequential(nn.Linear(3 + rr_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.delta = nn.Linear(hidden, 3)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def forward_windows(self, base_logits: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        tokens = self.input(torch.cat((base_logits, rr), dim=1))
        n = tokens.shape[0]; lengths = torch.minimum(torch.arange(1, n + 1, device=tokens.device),
                                                      torch.tensor(self.window, device=tokens.device))
        batch = tokens.new_zeros((n, self.window, tokens.shape[1]))
        for index in range(n):
            length = int(lengths[index]); start = index + 1 - length
            batch[index, :length] = tokens[start:index + 1]
        packed = nn.utils.rnn.pack_padded_sequence(batch, lengths.cpu(), batch_first=True,
                                                    enforce_sorted=False)
        _, state = self.gru(packed)
        return base_logits + self.max_correction * self.delta(state[-1])


class EmbeddingGRU(nn.Module):
    """Causal record state over the frozen current-beat morphology embedding.

    Unlike ``LogitGRU``, the historical state receives the 160-D feature before
    the CNN's three-class bottleneck.  It remains a residual classifier so a
    poorly estimated early history cannot erase the current-beat emission.
    """
    def __init__(self, embedding_dim: int = 160, rr_dim: int = 8, hidden: int = 128,
                 max_correction: float = .75) -> None:
        super().__init__()
        self.max_correction = float(max_correction)
        self.input = nn.Sequential(nn.Linear(embedding_dim + rr_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.delta = nn.Linear(hidden, 3)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward_chunk(self, base_logits: torch.Tensor, rr: torch.Tensor, embedding: torch.Tensor,
                      state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.input(torch.cat((embedding, rr), dim=1))
        history, new_state = self.gru(token.unsqueeze(0), state)
        return base_logits + self.max_correction * self.delta(history.squeeze(0)), new_state


class CausalViewGate(nn.Module):
    """Per-beat, label-free fusion of two already-causal single-lead streams.

    The gate sees only two current logits and the current causal RR descriptor.
    Its zero-initialized residual makes arithmetic logit averaging the initial
    behavior; ``max_correction`` bounds any supervised DS1 adjustment.
    """
    def __init__(self, rr_dim: int = 8, hidden: int = 32, max_correction: float = .25) -> None:
        super().__init__()
        self.max_correction = float(max_correction)
        self.body = nn.Sequential(nn.Linear(6 + rr_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, first_logits: torch.Tensor, second_logits: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        mean = (first_logits + second_logits) / 2.0
        return mean + self.max_correction * self.body(torch.cat((first_logits, second_logits, rr), dim=1))
