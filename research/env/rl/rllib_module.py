"""Masked, autoregressive, factored actor + centralized critic for RLlib PPO.

The action space (see `env/actions.py`) factors as

    (action_type, food, side, price, quantity)

and the policy samples autoregressively in that order. Each head's log-
probability contributes to the joint log-prob only when its dimension is
*relevant* for the sampled action type (Wait uses only `type`; Cancel uses
type+food+side; Take uses type+food+side+qty; Post uses all five).
Irrelevant heads contribute 0.

Sampling and log-prob computation live in `env.rl.factored_actor`; this
file wraps them in a torso (MLP or LSTM), a centralized critic, and the
RLlib RLModule interface. During training, `_forward_train` teacher-forces
the previously-sampled actions and returns the resulting log-prob and
entropy under `ACTION_DIST_INPUTS`. The paired `FactoredActionDistribution`
class is a thin proxy — it stores the precomputed tensors and just returns
them from `.logp()` and `.entropy()`.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule

# Note: we deliberately do NOT inherit from `ray.rllib.models.torch.
# torch_distributions.TorchDistribution`. That base class is a torch-Distribution
# wrapper for single-head categorical / normal / etc.; its `_get_torch_distribution`
# abstract method has no meaningful implementation for our autoregressive
# multi-head setup. RLlib's PPO learner only calls `.from_logits`, `.logp`,
# `.entropy`, and (when kl_coeff>0) `.kl` on the distribution — a plain class
# implementing those methods satisfies the actual contract.

from ..actions import NUM_ACTIONS
from .factored_actor import (
    DEFAULT_HEAD_KIND,
    FactoredActor,
    FactoredMaskTensors,
    factored_masks_from_flat_tensor,
    factored_to_flat_batch,
    flat_to_factored_batch,
)
from .global_observation_encoder import global_observation_dim
from .observation_encoder import observation_dim

ACTOR_HIDDEN_SIZE = 256
CRITIC_HIDDEN_SIZES = [512, 512, 256]
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 1
MAX_SEQ_LEN = 64


def _mlp(input_size: int, hidden_sizes: list[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    size = input_size
    for hidden_size in hidden_sizes:
        layers.extend(
            [
                nn.Linear(size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
            ]
        )
        size = hidden_size
    return nn.Sequential(*layers), size


class MLPTorso(nn.Module):
    """Feed-forward torso — the default for the initial IPPO/MAPPO baselines.

    Same OBS_DIM → ACTOR_HIDDEN_SIZE → ACTOR_HIDDEN_SIZE stack that the LSTM
    variant uses as its encoder + head, but with no recurrence. Interface
    matches `LSTMTorso.forward`: returns `(embedding, state_out=None)` and
    tolerates optional [B, T, F] inputs by folding the time axis for the MLP
    pass and unfolding on the way out.
    """

    def __init__(self, input_size: int = observation_dim()) -> None:
        super().__init__()
        self.net, self.output_size = _mlp(
            input_size, [ACTOR_HIDDEN_SIZE, ACTOR_HIDDEN_SIZE]
        )

    def get_initial_state(self) -> dict[str, np.ndarray]:
        # No recurrent state — return empty so RLlib doesn't try to plumb
        # `Columns.STATE_IN` / `STATE_OUT` for this actor.
        return {}

    def forward(
        self,
        observations: torch.Tensor,
        state_in: Optional[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, None]:
        del state_in  # unused; MLP is stateless
        squeeze_time = observations.ndim == 3
        if squeeze_time:
            B, T, F = observations.shape
            flat = observations.reshape(B * T, F)
            emb = self.net(flat).reshape(B, T, -1)
        else:
            emb = self.net(observations)
        return emb, None


class LSTMTorso(nn.Module):
    """Shared-torso variant that ends in an embedding fed to the factored
    head, not the final flat softmax layer."""

    def __init__(self, input_size: int = observation_dim()) -> None:
        super().__init__()
        self.input_size = input_size
        self.encoder, encoded_size = _mlp(input_size, [ACTOR_HIDDEN_SIZE])
        self.lstm = nn.LSTM(
            encoded_size,
            LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
        )
        self.head, self.output_size = _mlp(LSTM_HIDDEN_SIZE, [ACTOR_HIDDEN_SIZE])

    def get_initial_state(self) -> dict[str, np.ndarray]:
        return {
            "h": np.zeros(
                (LSTM_NUM_LAYERS, LSTM_HIDDEN_SIZE),
                dtype=np.float32,
            ),
            "c": np.zeros(
                (LSTM_NUM_LAYERS, LSTM_HIDDEN_SIZE),
                dtype=np.float32,
            ),
        }

    def forward(
        self,
        observations: torch.Tensor,
        state_in: Optional[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(state_in, dict):
            raise ValueError("LSTM torso requires Columns.STATE_IN")

        squeeze_time = observations.ndim == 2
        sequence = observations.unsqueeze(1) if squeeze_time else observations
        if sequence.ndim != 3:
            raise ValueError(
                f"LSTM observations must have shape [B, {self.input_size}] "
                f"or [B, T, {self.input_size}]"
            )

        if "h" not in state_in or "c" not in state_in:
            raise ValueError("LSTM torso requires h and c states")
        hidden = self._to_torch_state(state_in["h"])
        cell = self._to_torch_state(state_in["c"])
        embeddings, (hidden_out, cell_out) = self.lstm(
            self.encoder(sequence),
            (hidden, cell),
        )
        emb = self.head(embeddings)  # (B, T, ACTOR_HIDDEN_SIZE)
        if squeeze_time:
            emb = emb.squeeze(1)
        return emb, {
            "h": hidden_out.transpose(0, 1),
            "c": cell_out.transpose(0, 1),
        }

    @staticmethod
    def _to_torch_state(state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            state = state.unsqueeze(0)
        if state.ndim != 3:
            raise ValueError("LSTM state must have shape [B, L, H]")
        return state.transpose(0, 1).contiguous()


class MLPCritic(nn.Module):
    """Feed-forward critic. Input dim varies by algorithm:
    - MAPPO: `GLOBAL_OBS_DIM` (centralized rotated state, 305-D).
    - IPPO:  `OBS_DIM` (same local obs the actor sees, 171-D).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoder, output_size = _mlp(input_dim, CRITIC_HIDDEN_SIZES)
        self.value = nn.Linear(output_size, 1)

    def forward(self, critic_observations: torch.Tensor) -> torch.Tensor:
        return self.value(self.encoder(critic_observations)).squeeze(-1)


# Public defaults — override via `model_config["torso"]` / `model_config["algo"]`.
DEFAULT_TORSO = "mlp"
DEFAULT_ALGO = "mappo"


class FactoredActionDistribution:
    """RLlib-compatible proxy over the factored autoregressive actor.

    The actual sampling / log-prob math is done inside `NashHungerPPOTorchRLModule`
    which returns *already-computed* logp and entropy tensors in
    `ACTION_DIST_INPUTS`. This class just exposes them through the
    distribution interface that RLlib's PPO learner expects.

    That indirection is intentional. The autoregressive policy cannot be
    reconstructed from a fixed-shape logits tensor alone — computing logp
    requires re-running the conditioned heads against the network's
    parameters, which only the RLModule can do. Precomputing logp inside
    `_forward_train` and just returning it here keeps the correct
    gradient flow (through the RLModule → torso + heads) without asking
    RLlib to plumb network references through its distribution factory.
    """

    def __init__(self, inputs: dict[str, torch.Tensor]) -> None:
        # Deliberately NOT calling super().__init__() — TorchDistribution's
        # constructor expects a torch.distributions.Distribution which we
        # don't have (and don't need). All the methods we override below
        # are self-contained.
        self._logp = inputs["logp"]
        self._entropy = inputs["entropy"]
        self._actions = inputs.get("actions")  # only present at sample time

    @classmethod
    def from_logits(cls, logits: Any) -> "FactoredActionDistribution":
        return cls(logits)

    def sample(self, **kwargs) -> torch.Tensor:
        if self._actions is None:
            raise RuntimeError(
                "FactoredActionDistribution: no actions on this instance — "
                "sample() is only valid after a forward_exploration/inference "
                "pass. Training paths use logp() / entropy() directly."
            )
        return self._actions

    def rsample(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("factored discrete distribution is not reparameterized")

    def logp(self, actions: torch.Tensor) -> torch.Tensor:
        # actions parameter is redundant here — _forward_train has already
        # computed logp for exactly these actions (they came from the same
        # batch). Sanity-check the shape to catch upstream mismatches.
        if actions.shape != self._logp.shape:
            raise ValueError(
                f"actions shape {tuple(actions.shape)} does not match "
                f"precomputed logp shape {tuple(self._logp.shape)}"
            )
        return self._logp

    def entropy(self) -> torch.Tensor:
        """Single-sample estimate of the autoregressive joint entropy.

        Each row is `H(type) + Σ_i H(child_i | teacher-forced prefix from
        this row's sampled action)`. Averaged over a batch this is
        an unbiased estimator of the true joint autoregressive entropy;
        per row it is *stochastic* — it depends on the specific prefix
        this row was sampled from. If you need the exact joint entropy
        (e.g. for a tight entropy regularizer), you'd have to expand the
        expectation over all prefix values, which for our action-type
        cardinality of 4 (× food 4 × side 2 × price 30) is tractable but
        not what this method computes. PPO's entropy bonus tolerates the
        noisier single-sample estimator without issue.
        """
        return self._entropy

    def kl(self, other: "FactoredActionDistribution") -> torch.Tensor:
        # Computing exact KL for an autoregressive Categorical joint
        # requires marginalising each conditional head over every prefix
        # value — doable at our cardinalities but non-trivial, and we have
        # no consumer yet: PPO's default `kl_coeff=0` never calls this.
        # Returning `logp_new - logp_old` (a common one-sample estimator)
        # would silently plumb a wrong tensor into any downstream code
        # that assumes exact KL, so we fail loudly instead.
        raise NotImplementedError(
            "FactoredActionDistribution.kl() is not implemented. This "
            "policy uses an autoregressive factorization whose exact KL "
            "requires marginalising each head over previous samples. "
            "PPO with kl_coeff=0 does not need this. If you turn on "
            "kl_coeff or use a KL-early-stopping schedule, implement "
            "kl() properly here first — do not rely on a single-sample "
            "estimator masquerading as an exact KL."
        )

    @staticmethod
    def required_input_dim(space: gym.spaces.Space, **kwargs) -> int:
        del space, kwargs
        # We don't take flat logits; the RLModule fills a dict directly.
        return 0


class NashHungerPPOTorchRLModule(TorchRLModule, ValueFunctionAPI):
    def setup(self) -> None:
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("observation_space must be gymnasium.spaces.Dict")
        required_keys = {"observations", "action_mask", "critic_observations"}
        missing = required_keys.difference(self.observation_space.spaces)
        if missing:
            raise ValueError(
                f"observation_space is missing required keys: {sorted(missing)}"
            )
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("action_space must be gymnasium.spaces.Discrete")
        self.include_agent_id = bool(
            self.model_config.get("include_agent_id", False)
        )
        self.include_survival_features = bool(
            self.model_config.get("include_survival_features", False)
        )
        self.include_trade_history = bool(
            self.model_config.get("include_trade_history", False)
        )
        self._actor_obs_dim = observation_dim(
            include_agent_id=self.include_agent_id,
            include_survival_features=self.include_survival_features,
        )
        self._global_obs_dim = global_observation_dim(
            include_agent_id=self.include_agent_id,
            include_trade_history=self.include_trade_history,
        )
        if self.observation_space["observations"].shape != (self._actor_obs_dim,):
            raise ValueError(
                f"observations must have shape ({self._actor_obs_dim},), got "
                f"{self.observation_space['observations'].shape}"
            )
        if self.observation_space["action_mask"].shape != (NUM_ACTIONS,):
            raise ValueError(
                f"action_mask must have shape ({NUM_ACTIONS},), got "
                f"{self.observation_space['action_mask'].shape}"
            )
        if self.observation_space["critic_observations"].shape != (
            self._global_obs_dim,
        ):
            raise ValueError(
                f"critic_observations must have shape ({self._global_obs_dim},), got "
                f"{self.observation_space['critic_observations'].shape}"
            )
        if self.action_space.n != NUM_ACTIONS:
            raise ValueError(
                f"action space must contain {NUM_ACTIONS} actions, got "
                f"{self.action_space.n}"
            )

        # ---- torso ----
        torso_kind = self.model_config.get("torso", DEFAULT_TORSO)
        if torso_kind == "mlp":
            self.torso = MLPTorso(self._actor_obs_dim)
        elif torso_kind == "lstm":
            self.torso = LSTMTorso(self._actor_obs_dim)
        else:
            raise ValueError(
                f"unknown torso {torso_kind!r}; expected 'mlp' or 'lstm'"
            )
        self._is_recurrent = torso_kind == "lstm"

        head_kind = self.model_config.get("head_kind", DEFAULT_HEAD_KIND)
        self.actor = FactoredActor(
            embedding_size=self.torso.output_size,
            head_kind=head_kind,
        )

        # ---- critic (algorithm switch) ----
        # IPPO uses the same *local* observation as the actor for its value
        # function; MAPPO uses the centralized global-state view. This is
        # the ONLY structural difference between the two algos — actor,
        # reward, action mask, optimizer, and hyperparameters are shared.
        algo = self.model_config.get("algo", DEFAULT_ALGO).lower()
        if algo == "mappo":
            self._critic_obs_key = "critic_observations"
            critic_dim = self._global_obs_dim
        elif algo == "ippo":
            self._critic_obs_key = "observations"
            critic_dim = self._actor_obs_dim
        else:
            raise ValueError(
                f"unknown algo {algo!r}; expected 'ippo' or 'mappo'"
            )
        self._algo = algo
        self.critic = MLPCritic(input_dim=critic_dim)

    def get_initial_state(self) -> dict[str, np.ndarray]:
        return self.torso.get_initial_state()

    # ------------------------------------------------------------------
    # RLlib distribution class factories
    # ------------------------------------------------------------------

    def get_train_action_dist_cls(self) -> type:
        return FactoredActionDistribution

    def get_exploration_action_dist_cls(self) -> type:
        return FactoredActionDistribution

    def get_inference_action_dist_cls(self) -> type:
        return FactoredActionDistribution

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _observation_dict(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        observations = batch.get(Columns.OBS)
        if not isinstance(observations, dict):
            raise ValueError(
                f"{Columns.OBS!r} must contain a dictionary observation"
            )
        required = {"observations", "action_mask", "critic_observations"}
        missing = required.difference(observations)
        if missing:
            raise ValueError(
                f"observation batch is missing required keys: {sorted(missing)}"
            )
        return observations

    def _torso_and_masks(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, FactoredMaskTensors, Optional[dict[str, torch.Tensor]], tuple[int, ...]]:
        """Common preamble for all forward variants — runs the torso, folds
        any [B, T, ...] into [B*T, ...] for the actor, and derives per-head
        masks. Returns (emb, masks, state_out, shape_prefix) so callers can
        reshape their outputs back to the original prefix.
        """
        obs = self._observation_dict(batch)
        emb, state_out = self.torso(obs["observations"], batch.get(Columns.STATE_IN))
        flat_mask = obs["action_mask"]
        prefix = emb.shape[:-1]  # (B,) or (B, T)
        emb_flat = emb.reshape(-1, emb.shape[-1])
        mask_flat = flat_mask.reshape(-1, flat_mask.shape[-1])
        masks = factored_masks_from_flat_tensor(mask_flat)
        return emb_flat, masks, state_out, prefix

    # ------------------------------------------------------------------
    # Forward variants
    # ------------------------------------------------------------------

    def _forward_sample(
        self, batch: dict[str, Any], *, deterministic: bool
    ) -> dict[str, Any]:
        emb, masks, state_out, prefix = self._torso_and_masks(batch)
        factored, logp, entropy = self.actor.sample(
            emb, masks, deterministic=deterministic
        )
        actions_flat = factored_to_flat_batch(factored)  # (B*T,) long

        # Reshape back to whatever the caller sent in.
        actions = actions_flat.reshape(prefix)
        logp = logp.reshape(prefix)
        entropy = entropy.reshape(prefix)

        out: dict[str, Any] = {
            Columns.ACTIONS: actions,
            Columns.ACTION_LOGP: logp,
            Columns.ACTION_DIST_INPUTS: {
                "actions": actions,
                "logp": logp,
                "entropy": entropy,
            },
        }
        if state_out is not None:
            out[Columns.STATE_OUT] = state_out
        return out

    def _forward_inference(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        return self._forward_sample(batch, deterministic=True)

    def _forward_exploration(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        return self._forward_sample(batch, deterministic=False)

    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        return self._forward_exploration(batch, **kwargs)

    def _forward_train(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        emb, masks, state_out, prefix = self._torso_and_masks(batch)

        actions_batch = batch.get(Columns.ACTIONS)
        if actions_batch is None:
            raise ValueError(
                f"_forward_train requires {Columns.ACTIONS!r} in the batch"
            )
        actions_flat_ids = actions_batch.reshape(-1).long()
        factored = flat_to_factored_batch(actions_flat_ids)
        logp, entropy = self.actor.log_prob_and_entropy(emb, factored, masks)

        logp = logp.reshape(prefix)
        entropy = entropy.reshape(prefix)

        out: dict[str, Any] = {
            Columns.ACTION_DIST_INPUTS: {
                "actions": actions_batch,
                "logp": logp,
                "entropy": entropy,
            },
            Columns.ACTION_LOGP: logp,
        }
        if state_out is not None:
            out[Columns.STATE_OUT] = state_out
        return out

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> torch.Tensor:
        """IPPO reads the local observation; MAPPO reads the centralized
        observation. The key was picked in `setup()` based on `algo`."""
        del embeddings
        return self.critic(self._observation_dict(batch)[self._critic_obs_key])
