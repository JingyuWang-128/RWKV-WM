"""
JEPA+RWKV-6 World Model for Atari (Discrete Action Space)

This module implements the JEPA (Joint Embedding Predictive Architecture) with
RWKV-6 predictor for Atari environments with discrete actions.

Key differences from MuJoCo version:
- Discrete action space with one-hot encoding
- Integer action indices instead of continuous actions

Architecture:
- CNN Encoder + optional Self-Attention
- RWKV-6 Predictor (data-dependent decay)
- SIGReg for representation collapse prevention
- Optional lightweight decoder with exponential decay
- Optional EMA target encoder
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.networks as net
import modules.functions_losses as func

from common import SIGReg, RWKV6Predictor, SelfAttentionBlock, LightweightDecoder, TargetEncoderEMA


permute = lambda x: x.permute(0, 3, 1, 2)
swap = lambda x: torch.transpose(x, 0, 1)


class JEPARWKVWorldModel(nn.Module):
    """
    JEPA+RWKV-6 World Model for Atari (Discrete Actions).

    Uses pure deterministic JEPA latent space (no stochastic state).
    feat_dim = embed_dim (not stoch*discrete + hidden like PSSM).

    Args:
        video_log: Step interval for video logging
        obs_shape: Observation shape (H, W, C)
        num_action: Number of discrete actions
        embed_dim: JEPA embedding dimension
        hidden: Hidden dimension for heads
        num_rwkv_layers: Number of RWKV-6 layers
        n_heads: Number of attention heads in RWKV
        stem_ch: Stem channels for encoder
        min_res: Minimum resolution in encoder
        num_bin: Number of bins for two-hot encoding
        max_bin: Maximum bin value
        use_self_attention: Whether to use self-attention after encoder
        use_decoder: Whether to use reconstruction decoder
        decoder_weight: Initial weight for decoder loss
        decoder_decay: Decay rate for decoder weight
        sigreg_weight: Weight for SIGReg loss
        sigreg_knots: Knots for SIGReg
        sigreg_num_proj: Number of projections for SIGReg
        use_ema_target: Whether to use EMA target encoder
        ema_decay: EMA decay rate for target encoder
        gamma: Discount factor
        lambd: Lambda for GAE
        tau: Soft update rate
        lr: Learning rate
        eps: Adam epsilon
        use_amp: Whether to use automatic mixed precision
        act: Activation function class
        device: Device string
    """

    def __init__(self,
                 video_log,
                 obs_shape,
                 num_action,
                 embed_dim,
                 hidden,
                 num_rwkv_layers,
                 n_heads,
                 stem_ch,
                 min_res,
                 num_bin,
                 max_bin,
                 use_self_attention=True,
                 use_decoder=True,
                 decoder_weight=1.0,
                 decoder_decay=0.9999,
                 sigreg_weight=0.09,
                 sigreg_knots=17,
                 sigreg_num_proj=1024,
                 use_ema_target=False,
                 ema_decay=0.99,
                 gamma=0.997,
                 lambd=0.95,
                 tau=0.02,
                 lr=1e-4,
                 eps=1e-8,
                 use_amp=False,
                 act=nn.SiLU,
                 device="cuda",
                 ):
        super().__init__()

        # Store config
        self.num_action = num_action
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.feat_dim = embed_dim  # Pure deterministic JEPA: feat = embed
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.device = device
        self.video_log = video_log

        # Decoder config
        self.use_decoder = use_decoder
        self.initial_decoder_weight = decoder_weight
        self.decoder_decay = decoder_decay
        self.current_decoder_weight = decoder_weight

        # SIGReg config
        self.sigreg_weight = sigreg_weight

        # EMA config
        self.use_ema_target = use_ema_target
        self.ema_decay = ema_decay

        # AMP config
        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = use_amp

        # Imagination buffer
        self.batch_size = -1
        self.horizon = -1

        # ========== Build Encoder ==========
        self.encoder = net.Encoder(obs_shape[0], obs_shape[-1], stem_ch, min_res, act)
        encoder_out_dim = self.encoder.embed  # Flattened encoder output

        # Project encoder output to embed_dim
        self.encoder_proj = nn.Sequential(
            nn.Linear(encoder_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            act(),
        )

        # Optional self-attention after encoder
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.self_attention = SelfAttentionBlock(
                embed_dim, num_heads=n_heads, dropout=0.1)

        # ========== Build Target Encoder (optional EMA) ==========
        if use_ema_target:
            # Create a copy of encoder + projection for EMA
            self.target_encoder = net.Encoder(obs_shape[0], obs_shape[-1], stem_ch, min_res, act)
            self.target_encoder_proj = nn.Sequential(
                nn.Linear(encoder_out_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                act(),
            )
            if use_self_attention:
                self.target_self_attention = SelfAttentionBlock(
                    embed_dim, num_heads=n_heads, dropout=0.1)

            # Initialize with same weights
            self._copy_encoder_to_target()

            # Freeze target encoder
            for param in self.target_encoder.parameters():
                param.requires_grad = False
            for param in self.target_encoder_proj.parameters():
                param.requires_grad = False
            if use_self_attention:
                for param in self.target_self_attention.parameters():
                    param.requires_grad = False

        # ========== Build RWKV-6 Predictor ==========
        self.predictor = RWKV6Predictor(
            embed_dim=embed_dim,
            hidden=hidden,
            action_dim=num_action,  # One-hot action dimension
            num_layers=num_rwkv_layers,
            n_heads=n_heads,
            ffn_mult=4,
        )

        # ========== Build Decoder (optional) ==========
        if use_decoder:
            # Lightweight decoder for reconstruction
            self.decoder = LightweightDecoder(
                embed_dim=embed_dim,
                out_channels=obs_shape[-1],
                hidden_channels=stem_ch * 4,
                output_size=obs_shape[0],
            )

        # ========== Build Heads ==========
        self.reward_head = net.Head(embed_dim, num_bin, hidden, act)
        self.done_head = net.Head(embed_dim, 1, hidden, act)

        # ========== Build SIGReg ==========
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

        # ========== Loss Functions ==========
        self.mse_loss = func.MseLoss()
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin)
        self.bce_logits_loss = F.binary_cross_entropy_with_logits

        # ========== Optimizer ==========
        # Collect all trainable parameters
        params_list = []
        params_list.extend(self.encoder.parameters())
        params_list.extend(self.encoder_proj.parameters())
        if use_self_attention:
            params_list.extend(self.self_attention.parameters())
        params_list.extend(self.predictor.parameters())
        params_list.extend(self.reward_head.parameters())
        params_list.extend(self.done_head.parameters())
        if use_decoder:
            params_list.extend(self.decoder.parameters())

        self.optimizer = torch.optim.AdamW(params_list, lr=lr, eps=eps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Action encoding for discrete actions
        self.one_hot = lambda x: F.one_hot(x.long(), num_action).to(self.tensor_dtype)

    def _copy_encoder_to_target(self):
        """Copy online encoder weights to target encoder."""
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_encoder_proj.load_state_dict(self.encoder_proj.state_dict())
        if self.use_self_attention:
            self.target_self_attention.load_state_dict(self.self_attention.state_dict())

    @torch.no_grad()
    def _update_target_encoder(self):
        """EMA update for target encoder."""
        if not self.use_ema_target:
            return

        decay = self.ema_decay
        for target_param, online_param in zip(
            self.target_encoder.parameters(), self.encoder.parameters()
        ):
            target_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)

        for target_param, online_param in zip(
            self.target_encoder_proj.parameters(), self.encoder_proj.parameters()
        ):
            target_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)

        if self.use_self_attention:
            for target_param, online_param in zip(
                self.target_self_attention.parameters(), self.self_attention.parameters()
            ):
                target_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)

    def _get_decoder_weight(self, step):
        """Get current decoder weight with exponential decay."""
        return self.initial_decoder_weight * (self.decoder_decay ** step)

    def encode(self, obs, use_target=False):
        """
        Encode observations to JEPA embeddings.

        Args:
            obs: Observations of shape (B, T, C, H, W)
            use_target: Whether to use target encoder (for EMA mode)

        Returns:
            embed: Embeddings of shape (B, T, embed_dim)
        """
        if use_target and self.use_ema_target:
            # Use target encoder
            x = self.target_encoder(obs)  # (B, T, encoder_embed)
            x = self.target_encoder_proj(x)  # (B, T, embed_dim)
            if self.use_self_attention:
                x = self.target_self_attention(x)  # (B, T, embed_dim)
        else:
            # Use online encoder
            x = self.encoder(obs)  # (B, T, encoder_embed)
            x = self.encoder_proj(x)  # (B, T, embed_dim)
            if self.use_self_attention:
                x = self.self_attention(x)  # (B, T, embed_dim)

        return x

    @torch.no_grad()
    def preprocess(self, obs):
        """Preprocess numpy observations to tensor."""
        tensor_obs = torch.tensor(
            obs, dtype=self.tensor_dtype, device=self.device) / 255
        tensor_obs = tensor_obs.permute(0, 3, 1, 2)[:, None]  # [B, 1, C, H, W]
        return tensor_obs

    def initial(self, batch_size):
        """Get initial state for inference."""
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # Initialize RWKV state
            rwkv_state = self.predictor.initial_state(batch_size, self.device)

            # Initialize embedding (zeros)
            init_embed = torch.zeros(
                batch_size, self.embed_dim,
                dtype=self.tensor_dtype, device=self.device)

            return {
                "embed": init_embed,
                "rwkv_state": rwkv_state,
            }

    @torch.no_grad()
    def get_inference_feat(self, state, obs, is_first):
        """
        Get features for agent inference (single-step).

        Args:
            state: Previous state dict
            obs: Current observation (numpy)
            is_first: Whether this is the first step (numpy)

        Returns:
            feat: Features for agent (B, feat_dim)
            state: Updated state dict
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # Encode current observation
            embed = self.encode(self.preprocess(obs)).squeeze(1)  # (B, embed_dim)

            # Handle episode resets
            is_first_tensor = torch.tensor(is_first, dtype=self.tensor_dtype, device=self.device)
            if is_first_tensor.sum() > 0:
                init_state = self.initial(embed.shape[0])
                # Reset states for done environments
                weight = is_first_tensor.unsqueeze(-1)  # (B, 1)
                embed = embed * (1 - weight) + init_state["embed"] * weight

                # Reset RWKV states
                for key in state["rwkv_state"]:
                    init_val = init_state["rwkv_state"][key]
                    curr_val = state["rwkv_state"][key]
                    # Expand weight for state dimensions
                    num_extra_dims = curr_val.dim() - weight.dim()
                    weight_expanded = weight
                    for _ in range(num_extra_dims):
                        weight_expanded = weight_expanded.unsqueeze(-1)
                    state["rwkv_state"][key] = curr_val * (1 - weight_expanded) + init_val * weight_expanded

            state["embed"] = embed

        return embed, state

    @torch.no_grad()
    def update_inference_state(self, state, action):
        """
        Update state after action (single-step inference).

        Args:
            state: Current state dict
            action: Action taken (integer tensor, shape (B,))

        Returns:
            state: Updated state dict
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # One-hot encode action
            action_onehot = self.one_hot(action)  # (B, num_action)

            # Predict next embedding using RWKV (recurrent mode)
            embed = state["embed"].unsqueeze(1)  # (B, 1, embed_dim)
            action_onehot = action_onehot.unsqueeze(1)  # (B, 1, num_action)

            pred_embed, new_rwkv_state = self.predictor(
                embed, action_onehot,
                state=state["rwkv_state"],
                parallel=False
            )

            state["embed"] = pred_embed.squeeze(1)  # (B, embed_dim)
            state["rwkv_state"] = new_rwkv_state

        return state

    def init_imagine_buffer(self, batch_size, horizon):
        """Initialize buffers for imagination rollout."""
        if self.batch_size != batch_size or self.horizon != horizon:
            init_zeros = lambda s: torch.zeros(s, dtype=self.tensor_dtype, device=self.device)
            self.batch_size, self.horizon = batch_size, horizon

            embed_size = (batch_size, horizon + 1, self.embed_dim)
            action_size = (batch_size, horizon)

            self.embed_buffer = init_zeros(embed_size)
            self.action_buffer = init_zeros(action_size)

    @torch.no_grad()
    def get_video_frame(self, embed, index):
        """Generate video frame from embedding for visualization."""
        if self.use_decoder:
            pred_frame = self.decoder(embed[index, None])
            return pred_frame
        return None

    @torch.no_grad()
    def imagine_data(self, agent, obs, action, reward, done, is_first, horizon, logger=None, step=None):
        """
        Imagine future trajectories for agent training.

        Args:
            agent: Actor-critic agent
            obs: Observations (B, T, C, H, W)
            action: Actions (B, T) - integer indices
            reward: Rewards (B, T, 1)
            done: Done flags (B, T, 1)
            is_first: First step flags (B, T, 1)
            horizon: Imagination horizon
            logger: Logger for wandb
            step: Current training step

        Returns:
            feat: Imagined features (B*T, horizon+1, feat_dim)
            action: Imagined actions (B*T, horizon)
            discount: Discount factors (B*T, horizon)
            reward: Predicted rewards (B*T, horizon)
            weight: Importance weights (B*T, horizon)
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # Encode all observations
            embed = self.encode(obs)  # (B, T, embed_dim)

            # Flatten batch and time
            embed_flat = embed.flatten(0, 1)  # (B*T, embed_dim)
            batch_size = embed_flat.shape[0]

            self.init_imagine_buffer(batch_size, horizon)

            # Initialize RWKV state for imagination
            rwkv_state = self.predictor.initial_state(batch_size, self.device)

            video_index = torch.randint(batch_size, (1,), device=self.device)
            pred_video = []

            # Initial embedding
            current_embed = embed_flat

            for t in range(horizon):
                # Log video frame
                if logger is not None and step % self.video_log == 0 and self.use_decoder:
                    pred_video.append(self.get_video_frame(current_embed, video_index))

                # Store current embedding
                self.embed_buffer[:, t] = current_embed

                # Sample action from agent
                action_idx = agent.sample(current_embed)  # (B*T,) integer
                self.action_buffer[:, t] = action_idx.float()

                # One-hot encode action
                action_onehot = self.one_hot(action_idx)  # (B*T, num_action)

                # Predict next embedding
                current_embed_expanded = current_embed.unsqueeze(1)  # (B*T, 1, embed_dim)
                action_onehot_expanded = action_onehot.unsqueeze(1)  # (B*T, 1, num_action)

                pred_embed, rwkv_state = self.predictor(
                    current_embed_expanded, action_onehot_expanded,
                    state=rwkv_state, parallel=False
                )
                current_embed = pred_embed.squeeze(1)  # (B*T, embed_dim)

            # Store final embedding
            self.embed_buffer[:, -1] = current_embed

            # Compute rewards and dones from embeddings
            feat = self.embed_buffer
            discount = (self.done_head(feat[:, 1:]) < 0) * self.gamma
            reward = self.twohot_loss.decode(self.reward_head(feat[:, 1:]))
            weight = torch.cat((torch.ones_like(reward[:, :1]), discount[:, :-1]), dim=1)

        if logger is not None and step % self.video_log == 0 and len(pred_video) > 0:
            logger.log_video("Video/Imagination", torch.cat(pred_video, dim=1), step)

        return feat, self.action_buffer.long(), discount, reward, weight

    def update(self, agent, obs, action, reward, done, is_first, logger=None, step=None):
        """
        Update world model with a batch of data.

        Args:
            agent: Actor-critic agent (unused, for API compatibility)
            obs: Observations (B, T, C, H, W)
            action: Actions (B, T) - integer indices for discrete actions
            reward: Rewards (B, T, 1)
            done: Done flags (B, T, 1)
            is_first: First step flags (B, T, 1)
            logger: Logger for wandb
            step: Current training step
        """
        self.train()

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # ========== Encode observations ==========
            embed = self.encode(obs)  # (B, T, embed_dim)

            # ========== Get target embeddings ==========
            if self.use_ema_target:
                with torch.no_grad():
                    target_embed = self.encode(obs, use_target=True)
            else:
                # Stop-gradient on target
                target_embed = embed.detach()

            # ========== JEPA Prediction Loss ==========
            # One-hot encode actions for RWKV predictor
            action_onehot = self.one_hot(action)  # (B, T, num_action)

            # Predict next embeddings: pred[t] should match target[t+1]
            pred_embed, _ = self.predictor(
                embed[:, :-1], action_onehot[:, :-1],
                state=None, parallel=True
            )  # (B, T-1, embed_dim)

            # Prediction loss: MSE between predicted and target embeddings
            pred_loss = F.mse_loss(pred_embed, target_embed[:, 1:])

            # ========== SIGReg Loss ==========
            # SIGReg expects (T, B, D) format
            embed_for_sigreg = embed.transpose(0, 1)  # (T, B, embed_dim)
            sigreg_loss = self.sigreg(embed_for_sigreg)

            # ========== Reward and Done Prediction ==========
            reward_pred = self.reward_head(embed)
            done_pred = self.done_head(embed)

            reward_loss = self.twohot_loss(reward_pred, reward)
            done_loss = self.bce_logits_loss(done_pred, done)

            # ========== Decoder Loss (optional) ==========
            if self.use_decoder:
                # Reconstruct observations from embeddings
                recon = self.decoder(embed)  # (B, T, C, H, W)
                recon_loss = self.mse_loss(recon, obs)
                decoder_weight = self._get_decoder_weight(step if step is not None else 0)
            else:
                recon_loss = torch.tensor(0.0, device=self.device)
                decoder_weight = 0.0

            # ========== Total Loss ==========
            total_loss = (
                pred_loss +
                self.sigreg_weight * sigreg_loss +
                reward_loss +
                done_loss
            )
            if self.use_decoder:
                total_loss = total_loss + decoder_weight * recon_loss

        # ========== Backward and Optimize ==========
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # ========== Update Target Encoder (if EMA) ==========
        if self.use_ema_target:
            self._update_target_encoder()

        # ========== Logging ==========
        if logger is not None:
            logger.log("WorldModel/pred_loss", pred_loss.item(), step)
            logger.log("WorldModel/sigreg_loss", sigreg_loss.item(), step)
            logger.log("WorldModel/reward_loss", reward_loss.item(), step)
            logger.log("WorldModel/done_loss", done_loss.item(), step)
            if self.use_decoder:
                logger.log("WorldModel/recon_loss", recon_loss.item(), step)
                logger.log("WorldModel/decoder_weight", decoder_weight, step)

            # Log video
            if step % self.video_log == 0:
                video_index = torch.randint(obs.shape[0], (1,), device=self.device)
                logger.log_video("Video/Observation", obs[video_index], step)
                if self.use_decoder:
                    with torch.no_grad():
                        recon_video = self.decoder(embed[video_index])
                    logger.log_video("Video/Reconstruction", recon_video, step)
