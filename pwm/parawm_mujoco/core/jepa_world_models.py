"""
JEPA+RWKV World Model for MuJoCo tasks.

This implements a JEPA-style world model with RWKV-6 as the dynamics predictor.
Uses SIGReg for preventing representation collapse instead of KL divergence.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

import sys
sys.path.append('..')

from common.sigreg import SIGReg
from common.rwkv6 import RWKV6Predictor
from common.jepa_modules import (
    SelfAttentionBlock,
    LightweightDecoder,
    TargetEncoderEMA,
    DecayScheduler,
)

import modules.networks as net
import modules.functions_losses as func


params = lambda x: list(x.parameters())


class JEPARWKVWorldModel(nn.Module):
    """
    JEPA-style World Model with RWKV-6 dynamics predictor.

    Architecture:
    - Online Encoder (CNN + optional self-attention)
    - Target Encoder (EMA of online encoder, optional)
    - RWKV-6 Predictor (replaces PSSM)
    - Lightweight Decoder (optional, for physical consistency)
    - Reward Head (MLP)
    - Done Head (MLP)
    - SIGReg (for preventing representation collapse)

    Key differences from original ParallelWorldModel:
    - Deterministic state representation (no stochastic state)
    - JEPA-style prediction loss (MSE in latent space)
    - SIGReg regularization instead of KL divergence

    Args:
        video_log: Interval for video logging
        is_proprio: Whether using proprioceptive observations
        obs_shape: Observation shape
        action_dim: Action dimension
        embed_dim: Embedding dimension
        hidden: Hidden dimension for RWKV
        num_rwkv_layers: Number of RWKV layers
        n_heads: Number of attention heads
        stem_ch: Stem channel count for CNN
        min_res: Minimum resolution for CNN
        num_bin: Number of bins for reward prediction
        max_bin: Maximum bin value
        use_self_attention: Whether to use self-attention after CNN
        use_decoder: Whether to use reconstruction decoder
        decoder_weight: Initial weight for decoder loss
        decoder_decay: Decay rate for decoder loss weight
        sigreg_weight: Weight for SIGReg loss
        sigreg_knots: Number of knots for SIGReg
        sigreg_num_proj: Number of projections for SIGReg
        use_ema_target: Whether to use EMA target encoder
        ema_decay: Decay rate for EMA target encoder
        gamma: Discount factor
        lambd: Lambda for GAE
        tau: Soft update coefficient
        lr: Learning rate
        eps: Adam epsilon
        use_amp: Whether to use automatic mixed precision
        act: Activation function
        device: Device to use
    """

    def __init__(
        self,
        video_log: int,
        is_proprio: bool,
        obs_shape: tuple,
        action_dim: int,
        embed_dim: int,
        hidden: int,
        num_rwkv_layers: int,
        n_heads: int,
        stem_ch: int,
        min_res: int,
        num_bin: int,
        max_bin: int,
        use_self_attention: bool,
        use_decoder: bool,
        decoder_weight: float,
        decoder_decay: float,
        sigreg_weight: float,
        sigreg_knots: int,
        sigreg_num_proj: int,
        use_ema_target: bool,
        ema_decay: float,
        gamma: float,
        lambd: float,
        tau: float,
        lr: float,
        eps: float,
        use_amp: bool,
        act,
        device: str,
    ):
        super().__init__()

        # Store config
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.feat_dim = embed_dim  # Pure deterministic, no stochastic state

        # Loss weights
        self.sigreg_weight = sigreg_weight
        self.use_decoder = use_decoder
        self.use_ema_target = use_ema_target
        self.ema_decay = ema_decay

        # Training params
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.device = device
        self.video_log = video_log
        self.is_proprio = is_proprio

        # AMP settings
        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = use_amp

        # Decoder decay scheduler
        self.decoder_scheduler = DecayScheduler(
            initial_weight=decoder_weight,
            decay_rate=decoder_decay,
            min_weight=0.0,
        )
        self.train_step = 0

        # Imagination buffer
        self.batch_size = -1
        self.horizon = -1

        # Build encoder
        if is_proprio:
            num_layer, encode_dim = 3, hidden * 2
            self.encoder = net.ProprioEncoder(obs_shape, encode_dim, num_layer, act)
            encoder_embed = encode_dim
        else:
            self.encoder = net.Encoder(obs_shape[0], obs_shape[-1], stem_ch, min_res, act)
            encoder_embed = self.encoder.embed

        # Optional self-attention after CNN
        self.use_self_attention = use_self_attention and not is_proprio
        if self.use_self_attention:
            self.self_attention = SelfAttentionBlock(
                dim=encoder_embed,
                num_heads=n_heads,
                dropout=0.1,
            )

        # Projection to embed_dim if needed
        if encoder_embed != embed_dim:
            self.embed_proj = nn.Sequential(
                nn.Linear(encoder_embed, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.embed_proj = nn.Identity()

        # Target encoder (optional EMA)
        if use_ema_target:
            # Create a copy for EMA
            self.target_encoder = copy.deepcopy(self.encoder)
            for param in self.target_encoder.parameters():
                param.requires_grad = False

            if self.use_self_attention:
                self.target_self_attention = copy.deepcopy(self.self_attention)
                for param in self.target_self_attention.parameters():
                    param.requires_grad = False

            if encoder_embed != embed_dim:
                self.target_embed_proj = copy.deepcopy(self.embed_proj)
                for param in self.target_embed_proj.parameters():
                    param.requires_grad = False
            else:
                self.target_embed_proj = nn.Identity()

        # RWKV-6 Predictor
        self.predictor = RWKV6Predictor(
            embed_dim=embed_dim,
            hidden=hidden,
            action_dim=action_dim,
            num_layers=num_rwkv_layers,
            n_heads=n_heads,
            ffn_mult=4,
        )

        # Optional decoder for reconstruction
        if use_decoder and not is_proprio:
            self.decoder = net.Decoder(
                embed_dim, self.encoder.out_ch, obs_shape[-1], stem_ch, min_res, act
            )
        elif use_decoder and is_proprio:
            self.decoder = net.ProprioDecoder(
                embed_dim, obs_shape, hidden * 2, 3, act
            )
        else:
            self.decoder = None

        # Prediction heads
        self.reward_head = net.Head(embed_dim, num_bin, hidden, act)
        self.done_head = net.Head(embed_dim, 1, hidden, act)

        # SIGReg for preventing collapse
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

        # Loss functions
        self.mse_loss = func.MseLoss(is_proprio)
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin)
        self.bce_logits_loss = F.binary_cross_entropy_with_logits

        # Optimizer
        model_params = (
            params(self.encoder) +
            params(self.predictor) +
            params(self.reward_head) +
            params(self.done_head)
        )
        if self.use_self_attention:
            model_params += params(self.self_attention)
        if hasattr(self, 'embed_proj') and isinstance(self.embed_proj, nn.Sequential):
            model_params += params(self.embed_proj)
        if self.decoder is not None:
            model_params += params(self.decoder)

        self.optimizer = torch.optim.AdamW(model_params, lr=lr, eps=eps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to embeddings.

        Args:
            obs: Observations of shape (B, T, ...) or (B, ...)

        Returns:
            Embeddings of shape (B, T, embed_dim) or (B, embed_dim)
        """
        embed = self.encoder(obs)

        if self.use_self_attention and not self.is_proprio:
            # Apply self-attention
            has_time = embed.dim() == 3
            if has_time:
                B, T, D = embed.shape
                embed = embed.reshape(B * T, D)

            embed = embed.unsqueeze(1)  # (B*T, 1, D)
            embed = self.self_attention(embed)
            embed = embed.squeeze(1)  # (B*T, D)

            if has_time:
                embed = embed.reshape(B, T, D)

        embed = self.embed_proj(embed)
        return embed

    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observations using target encoder (EMA or stop-gradient).

        Args:
            obs: Observations of shape (B, T, ...) or (B, ...)

        Returns:
            Target embeddings of shape (B, T, embed_dim) or (B, embed_dim)
        """
        if self.use_ema_target:
            embed = self.target_encoder(obs)

            if self.use_self_attention and not self.is_proprio:
                has_time = embed.dim() == 3
                if has_time:
                    B, T, D = embed.shape
                    embed = embed.reshape(B * T, D)

                embed = embed.unsqueeze(1)
                embed = self.target_self_attention(embed)
                embed = embed.squeeze(1)

                if has_time:
                    embed = embed.reshape(B, T, D)

            embed = self.target_embed_proj(embed)
        else:
            # Use online encoder with stop-gradient
            with torch.no_grad():
                embed = self.encode(obs)

        return embed

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder with EMA of online encoder."""
        if not self.use_ema_target:
            return

        for target_param, online_param in zip(
            self.target_encoder.parameters(),
            self.encoder.parameters()
        ):
            target_param.data.mul_(self.ema_decay).add_(
                online_param.data, alpha=1 - self.ema_decay
            )

        if self.use_self_attention:
            for target_param, online_param in zip(
                self.target_self_attention.parameters(),
                self.self_attention.parameters()
            ):
                target_param.data.mul_(self.ema_decay).add_(
                    online_param.data, alpha=1 - self.ema_decay
                )

        if hasattr(self, 'target_embed_proj') and isinstance(self.target_embed_proj, nn.Sequential):
            for target_param, online_param in zip(
                self.target_embed_proj.parameters(),
                self.embed_proj.parameters()
            ):
                target_param.data.mul_(self.ema_decay).add_(
                    online_param.data, alpha=1 - self.ema_decay
                )

    @torch.no_grad()
    def preprocess(self, obs):
        """Preprocess observations."""
        if self.is_proprio:
            tensor_obs = torch.tensor(obs, dtype=self.tensor_dtype, device=self.device)
        else:
            tensor_obs = torch.tensor(obs, dtype=self.tensor_dtype, device=self.device) / 255
            tensor_obs = tensor_obs.permute(0, 3, 1, 2)[:, None]  # [B, 1, C, H, W]
        return tensor_obs

    @torch.no_grad()
    def get_inference_feat(self, state: Dict, obs, is_first) -> Tuple[torch.Tensor, Dict]:
        """
        Get feature for inference (agent action selection).

        Args:
            state: Previous state dict
            obs: Current observation
            is_first: Whether this is the first step

        Returns:
            feat: Feature tensor of shape (B, feat_dim)
            state: Updated state dict
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # Encode observation
            embed = self.encode(self.preprocess(obs)).squeeze(1)  # (B, embed_dim)

            # Handle episode reset
            is_first = torch.tensor(is_first, dtype=self.tensor_dtype, device=self.device)
            if is_first.sum() > 0:
                init_state = self.initial(embed.shape[0])
                for key, val in state.items():
                    num_axis = val.dim() - is_first.dim()
                    weight = is_first.unflatten(-1, [-1] + [1 for _ in range(num_axis)])
                    state[key] = val * (1 - weight) + init_state[key] * weight

            # Update state with current embedding
            state['embed'] = embed

        return embed, state

    @torch.no_grad()
    def update_inference_state(self, state: Dict, action) -> Dict:
        """
        Update state for next inference step.

        Args:
            state: Current state dict
            action: Action taken

        Returns:
            Updated state dict
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            embed = state['embed']
            action_tensor = torch.tensor(action, dtype=self.tensor_dtype, device=self.device)

            # Get RWKV state
            rwkv_state = state.get('rwkv_state', None)

            # Predict next embedding
            pred_embed, new_rwkv_state = self.predictor(
                embed, action_tensor,
                state=rwkv_state,
                parallel=False
            )

            state['embed'] = pred_embed
            state['rwkv_state'] = new_rwkv_state

        return state

    def initial(self, batch_size: int) -> Dict:
        """
        Create initial state.

        Args:
            batch_size: Batch size

        Returns:
            Initial state dict
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            return {
                'embed': torch.zeros(batch_size, self.embed_dim, device=self.device, dtype=self.tensor_dtype),
                'rwkv_state': self.predictor.initial_state(batch_size, self.device),
            }

    def init_imagine_buffer(self, batch_size: int, horizon: int):
        """Initialize imagination buffer."""
        if self.batch_size != batch_size or self.horizon != horizon:
            init_zeros = lambda s: torch.zeros(s, dtype=self.tensor_dtype, device=self.device)
            self.batch_size, self.horizon = batch_size, horizon

            embed_size = (batch_size, horizon + 1, self.embed_dim)
            action_size = (batch_size, horizon, self.action_dim)
            self.embed_buffer = init_zeros(embed_size)
            self.action_buffer = init_zeros(action_size)

    @torch.no_grad()
    def get_video_frame(self, embed: torch.Tensor, index: int) -> torch.Tensor:
        """Get reconstructed video frame for logging."""
        if self.decoder is not None:
            return self.decoder(embed[index, None])
        return None

    @torch.no_grad()
    def imagine_data(
        self,
        agent,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        is_first: torch.Tensor,
        horizon: int,
        logger=None,
        step=None
    ):
        """
        Generate imagination rollouts for agent training.

        Args:
            agent: Actor-critic agent
            obs: Observations (B, T, ...)
            action: Actions (B, T, action_dim)
            reward: Rewards (B, T, 1)
            done: Done flags (B, T, 1)
            is_first: Is-first flags (B, T, 1)
            horizon: Imagination horizon
            logger: Logger for video
            step: Current step

        Returns:
            feat: Features (B*T, horizon+1, embed_dim)
            action: Actions (B*T, horizon, action_dim)
            discount: Discounts (B*T, horizon, 1)
            reward: Rewards (B*T, horizon, 1)
            weight: Weights (B*T, horizon, 1)
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # Encode observations
            embed = self.encode(obs)  # (B, T, embed_dim)

            # Flatten batch and time
            img_embed = embed.flatten(0, 1)  # (B*T, embed_dim)
            batch_size = img_embed.shape[0]
            self.init_imagine_buffer(batch_size, horizon)

            video_index = torch.randint(batch_size, (1,), device=self.device)
            pred_video = []

            # Initialize RWKV state
            rwkv_state = self.predictor.initial_state(batch_size, self.device)

            for t in range(horizon):
                if logger is not None and not self.is_proprio:
                    if step % self.video_log == 0:
                        frame = self.get_video_frame(img_embed, video_index)
                        if frame is not None:
                            pred_video.append(frame)

                self.embed_buffer[:, t] = img_embed
                self.action_buffer[:, t] = agent.sample(img_embed)

                # Predict next embedding
                pred_embed, rwkv_state = self.predictor(
                    img_embed, self.action_buffer[:, t],
                    state=rwkv_state,
                    parallel=False
                )
                img_embed = pred_embed

            self.embed_buffer[:, -1] = img_embed

            # Compute discount and reward from embeddings
            feat = self.embed_buffer
            discount = (self.done_head(self.embed_buffer[:, 1:]) < 0) * self.gamma
            reward = self.twohot_loss.decode(self.reward_head(self.embed_buffer[:, 1:]))
            weight = torch.cat((torch.ones_like(reward[:, :1]), discount[:, :-1]), dim=1)

        if logger is not None and not self.is_proprio:
            if step % self.video_log == 0 and len(pred_video) > 0:
                logger.log_video("Video/Imagination", torch.cat(pred_video, dim=1), step)

        return feat, self.action_buffer, discount, reward, weight

    def update(
        self,
        agent,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        is_first: torch.Tensor,
        logger=None,
        step=None
    ):
        """
        Training step for JEPA+RWKV world model.

        Args:
            agent: Actor-critic agent (unused, for API compatibility)
            obs: Observations (B, T, ...)
            action: Actions (B, T, action_dim)
            reward: Rewards (B, T, 1)
            done: Done flags (B, T, 1)
            is_first: Is-first flags (B, T, 1)
            logger: Logger
            step: Current step
        """
        self.train()
        self.train_step += 1

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # Encode observations with online encoder
            embed = self.encode(obs)  # (B, T, embed_dim)

            # Get target embeddings (EMA or stop-gradient)
            target_embed = self.encode_target(obs)  # (B, T, embed_dim)

            # RWKV prediction: predict next embedding from current embedding and action
            # Input: embed[:, :-1], action[:, :-1]
            # Target: target_embed[:, 1:]
            pred_embed, _ = self.predictor(
                embed[:, :-1], action[:, :-1],
                parallel=True
            )

            # JEPA prediction loss
            pred_loss = F.mse_loss(pred_embed, target_embed[:, 1:].detach())

            # SIGReg loss (prevents collapse)
            sigreg_loss = self.sigreg(embed.transpose(0, 1))  # (T, B, D)

            # Reward and done prediction from embeddings
            reward_hat = self.reward_head(embed[:, 1:])
            done_hat = self.done_head(embed[:, 1:])

            reward_loss = self.twohot_loss(reward_hat, reward[:, 1:])
            done_loss = self.bce_logits_loss(done_hat, done[:, 1:])

            # Optional decoder loss with weight decay
            if self.decoder is not None and self.use_decoder:
                decoder_weight = self.decoder_scheduler.get_weight(self.train_step)
                obs_hat = self.decoder(embed)
                recon_loss = self.mse_loss(obs_hat, obs)
            else:
                recon_loss = torch.tensor(0.0, device=self.device)
                decoder_weight = 0.0

            # Total loss
            total_loss = (
                pred_loss +
                self.sigreg_weight * sigreg_loss +
                reward_loss +
                done_loss +
                decoder_weight * recon_loss
            )

        # Backward pass
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # Update target encoder (if using EMA)
        self.update_target_encoder()

        # Logging
        if logger is not None:
            logger.log("WorldModel/pred_loss", pred_loss.item(), step)
            logger.log("WorldModel/sigreg_loss", sigreg_loss.item(), step)
            logger.log("WorldModel/reward_loss", reward_loss.item(), step)
            logger.log("WorldModel/done_loss", done_loss.item(), step)
            if self.use_decoder:
                logger.log("WorldModel/recon_loss", recon_loss.item(), step)
                logger.log("WorldModel/decoder_weight", decoder_weight, step)

            if step % self.video_log == 0 and not self.is_proprio and self.decoder is not None:
                video_index = torch.randint(obs.shape[0], (1,), device=self.device)
                logger.log_video("Video/Observation", obs[video_index], step)
                obs_hat_video = self.decoder(embed)
                logger.log_video("Video/Reconstruction", obs_hat_video[video_index], step)
