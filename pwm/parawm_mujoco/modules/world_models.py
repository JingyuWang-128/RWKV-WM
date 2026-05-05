import copy
import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import OneHotCategorical

import modules.functions_losses as func
import modules.parallel_rnns as rnn
import modules.networks as net

# Add parent directory to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.sigreg import SIGReg, DecoderWeightScheduler
from modules.rwkv_predictor import RWKVPredictor, ContinuousActionEncoder
from modules.self_attention import SelfAttentionBlock, TemporalSelfAttention


params = lambda x: list(x.parameters())
permute = lambda x: x.permute(0, 3, 1, 2)
swap = lambda x: torch.transpose(x, 0, 1) 
ste_sample = lambda d: d.probs + (d.sample() - d.probs).detach()
to_param = lambda x: nn.Parameter(x)


class ParallelWorldModel(nn.Module):
    def __init__(self,
                 video_log,
                 is_proprio,
                 obs_shape,
                 action_dim,
                 stoch,
                 discrete,
                 hidden,
                 stem_ch,
                 min_res,
                 num_bin,
                 max_bin,
                 dyn_scale,
                 rep_scale,
                 val_scale,
                 kl_free,
                 gamma,
                 lambd,
                 tau,
                 lr,
                 eps,
                 use_amp,
                 act,
                 device,
                ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden = hidden
        self.stoch_dim = stoch * discrete
        self.feat_dim = self.stoch_dim + hidden
        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.val_scale = val_scale
        self.kl_free = kl_free
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.device = device
        self.batch_size = -1
        self.horizon = -1
        self.video_log = video_log
        self.is_proprio = is_proprio

        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = use_amp

        if is_proprio:
            num_layer, encode_dim = 3, hidden * 2
            self.encoder = net.ProprioEncoder(obs_shape, encode_dim, num_layer, act)
            self.decoder = net.ProprioDecoder(self.stoch_dim, obs_shape, encode_dim, num_layer, act)
        else:
            self.encoder = net.Encoder(obs_shape[0], obs_shape[-1], stem_ch, min_res, act)
            self.decoder = net.Decoder(
                self.stoch_dim, self.encoder.out_ch, obs_shape[-1], stem_ch, min_res, act)
        
        self.dynamic = PSSM(stoch, hidden, discrete, action_dim, self.encoder.embed, act, device)
        self.done_head = net.Head(hidden, 1, hidden, act)
        self.reward_head = net.Head(hidden, num_bin, hidden, act)

        self.mse_loss = func.MseLoss(is_proprio)
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin)
        self.bce_logits_loss = F.binary_cross_entropy_with_logits

        model_params = params(self.dynamic) + params(self.done_head) + params(self.reward_head)
        vae_params = params(self.encoder) + params(self.decoder)

        self.optimizer = torch.optim.AdamW(model_params + vae_params, lr=lr, eps=eps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def preprocess(self, obs):
        if self.is_proprio:
            tensor_obs = torch.tensor(obs, dtype=self.tensor_dtype, device=self.device)
        else:
            tensor_obs = torch.tensor(obs, dtype=self.tensor_dtype, device=self.device) / 255
            tensor_obs = tensor_obs.permute(0, 3, 1, 2)[:, None] # [B, 1, C, H, W]
        return tensor_obs

    @torch.no_grad()
    def get_inference_feat(self, state, obs, is_first):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            embed = self.encoder(self.preprocess(obs)).squeeze(1)
            obs_stats = self.dynamic.suff_stats_layer("obs", embed)
            obs_stoch = ste_sample(self.dynamic.get_dist(obs_stats))

            is_first = torch.tensor(is_first, dtype=self.tensor_dtype, device=self.device)
            if is_first.sum() > 0:
                init_state = self.initial(obs_stoch.shape[0])
                for key, val in state.items():
                    num_axis = val.dim() - is_first.dim()
                    weight = is_first.unflatten(-1, [-1] + [1 for _ in range(num_axis)])
                    state[key] = val * (1 - weight) + init_state[key] * weight

            state.update({"stoch": obs_stoch, **obs_stats})
        return self.dynamic.get_feat(state), state
    
    @torch.no_grad()
    def update_inference_state(self, state, action):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            img_step_stats = self.dynamic.img_step(state, action, True)
            deter, _, para_stats, _ = img_step_stats
            state.update({"deter": deter, **para_stats})
        return state

    def initial(self, batch_size):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            return self.dynamic.initial(batch_size)
    
    def init_imagine_buffer(self, batch_size, horizon):
        if self.batch_size != batch_size or self.horizon != horizon:
            init_zeros = lambda s: torch.zeros(s, dtype=self.tensor_dtype, device=self.device)
            self.batch_size, self.horizon = batch_size, horizon

            deter_size = (batch_size, horizon+1, self.hidden)
            stoch_size = (batch_size, horizon+1, self.stoch_dim)
            action_size = (batch_size, horizon, self.action_dim)
            self.deter_buffer = init_zeros(deter_size)
            self.stoch_buffer = init_zeros(stoch_size)
            self.action_buffer = init_zeros(action_size)
    
    @torch.no_grad()
    def get_video_frame(self, prior, index):
        stoch = self.dynamic.get_flatten_stoch(prior)
        pred_frame = self.decoder(stoch[index, None])
        return pred_frame

    @torch.no_grad()
    def imagine_data(self, agent, obs, action, reward, done, is_first, horizon, logger=None, step=None):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            state, _, _, _ = self.dynamic.parallel_observe(self.encoder(obs), action, is_first)
            img_state = {k: v.flatten(0, 1) for k, v in state.items()}
            batch_size = self.dynamic.get_feat(img_state).shape[0]
            self.init_imagine_buffer(batch_size, horizon)

            video_index, pred_video = torch.randint(batch_size, (1,), device=self.device), []

            for t in range(horizon):
                if logger is not None and not self.is_proprio:
                    if step % self.video_log == 0:
                        pred_video += [self.get_video_frame(img_state, video_index)]
    
                self.deter_buffer[:, t] = self.dynamic.get_deter(img_state)
                self.stoch_buffer[:, t] = self.dynamic.get_flatten_stoch(img_state)
                self.action_buffer[:, t] = agent.sample(
                    torch.cat((self.deter_buffer[:, t], self.stoch_buffer[:, t]), dim=-1))
                img_state = self.dynamic.img_step(img_state, self.action_buffer[:, t])
            
            self.deter_buffer[:, -1] = self.dynamic.get_deter(img_state)
            self.stoch_buffer[:, -1] = self.dynamic.get_flatten_stoch(img_state)

            feat = torch.cat((self.deter_buffer, self.stoch_buffer), dim=-1)
            discount = (self.done_head(self.deter_buffer[:, 1:]) < 0) * self.gamma
            reward = self.twohot_loss.decode(self.reward_head(self.deter_buffer[:, 1:]))
            weight = torch.cat((torch.ones_like(reward[:, :1]), discount[:, :-1]), dim=1)
            
        if logger is not None and not self.is_proprio:
            if step % self.video_log == 0:
                logger.log_video("Video/Imagination", torch.cat(pred_video, dim=1), step)

        return feat, self.action_buffer, discount, reward, weight

    def update(self, agent, obs, action, reward, done, is_first, logger=None, step=None):
        self.train()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            post, prior, stoch, deter = self.dynamic.parallel_observe(self.encoder(obs), action, is_first)
            dyn_loss, rep_loss, real_kl, ent = self.dynamic.kl_loss(post, prior, self.kl_free)

            obs_hat = self.decoder(stoch)
            done_hat = self.done_head(deter)
            reward_hat = self.reward_head(deter)

            recon_loss = self.mse_loss(obs_hat, obs)
            done_loss = self.bce_logits_loss(done_hat, done)
            reward_loss = self.twohot_loss(reward_hat, reward)
            
            head_loss = done_loss + reward_loss
            model_loss = self.dyn_scale * dyn_loss + head_loss
            vae_loss = recon_loss + self.rep_scale * rep_loss

        self.scaler.scale(model_loss + vae_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/recon_loss", recon_loss.item(), step)
            logger.log("WorldModel/reward_loss", reward_loss.item(), step)
            logger.log("WorldModel/dyn_loss", dyn_loss.item(), step)
            logger.log("WorldModel/rep_loss", rep_loss.item(), step)
            logger.log("WorldModel/real_kl", real_kl.item(), step)
            logger.log("WorldModel/vae_ent", ent.item(), step)

            if step % self.video_log == 0 and not self.is_proprio:
                video_index = torch.randint(obs.shape[0], (1,), device=self.device)
                logger.log_video("Video/Observation", obs[video_index], step)
                logger.log_video("Video/Reconstruction", obs_hat[video_index], step)


class PSSM(nn.Module):
    def __init__(self, stoch, hidden, discrete, action_dim, embed, act, device, unimix_ratio=0.01):
        super().__init__()
        self.stoch = stoch
        self.hidden = hidden
        self.discrete = discrete
        self.action_dim = action_dim
        self.unimix_ratio = unimix_ratio
        self.embed = embed
        self.act = act
        self.device = device
        self.num_rnns = 2

        stoch_dim = stoch * discrete
        inp_dim = stoch_dim + action_dim

        self.rnn_layer = self.init_cell()
        self.inp_layer = net.InpLayer(inp_dim, hidden, hidden, act)
        self.ims_stat_layer = net.ImsStatLayer(hidden, stoch_dim, act)
        self.obs_stat_layer = net.ObsStatLayer(embed, stoch_dim, act)
        
        cell_ws = {}
        for id in range(self.num_rnns):
            cell_stats = self.rnn_layer[id].initial(1, id)
            cell_stats = {k: v.to(device) for k, v in cell_stats.items()}
            cell_ws.update(cell_stats)
        self.cell_ws = cell_ws
        self.init_deter = torch.zeros(1, hidden, requires_grad=False).to(device)
    
    def init_cell(self):
        layer_list = []
        for i in range(self.num_rnns):
            layer_list += [rnn.RNNCell(self.hidden, self.hidden, self.act)]
        layers = nn.ModuleList(layer_list)
        return layers

    @torch.no_grad()
    def initial(self, batch_size):
        init = {k: v.expand(batch_size, v.shape[-1]) 
                for k, v in self.cell_ws.items()}
        init_deter = self.init_deter.expand(
            batch_size, self.init_deter.shape[-1])
        init_logit, init_stoch = self.get_init_stoch(init_deter)
        init.update({
            "logit": init_logit,
            "stoch": init_stoch, 
            "deter": init_deter, 
        })
        return init
    
    def get_init_stoch(self, deter):
        stats = self.suff_stats_layer("ims", deter)
        dist = self.get_dist(stats)
        return stats["logit"], dist.mode
    
    def get_deter(self, state):
        return state["deter"]
    
    def get_feat(self, state):
        stoch = state["stoch"].flatten(-2, -1)
        return torch.cat((state["deter"], stoch), dim=-1)
    
    def get_flatten_stoch(self, state):
        return state["stoch"].flatten(-2, -1)
    
    def get_dist(self, state):
        probs = F.softmax(state["logit"], dim=-1)
        probs = probs * (1 - self.unimix_ratio) + \
            self.unimix_ratio / self.discrete
        return OneHotCategorical(probs=probs)
    
    def parallel_observe(self, embed, action, is_first):
        init = self.initial(action.shape[0])
        obs_stats = self.suff_stats_layer("obs", embed)
        oracle_stoch = ste_sample(self.get_dist(obs_stats))

        flatten_stoch = oracle_stoch.flatten(-2, -1)
        concat_input = torch.cat((flatten_stoch, action), dim=-1)
        latent, mask = self.inp_layer(concat_input), is_first
        deter, para_stats = self.cell_layers(latent, init, mask, True)

        ims_stats = self.suff_stats_layer("ims", deter[:, :-1])
        ims_stoch = ste_sample(self.get_dist(ims_stats))

        obs_stats = {k: v[:, 1:] for k, v in obs_stats.items()}
        obs_stoch = oracle_stoch[:, 1:]

        stats = {"deter": deter, **para_stats}
        stats = {k: v[:, :-1] for k, v in stats.items()}
        post = {"stoch": obs_stoch, **obs_stats, **stats}
        prior = {"stoch": ims_stoch, **ims_stats, **stats}
        return post, prior, flatten_stoch, deter

    def img_step(self, prev_state, prev_action, return_stats=False):
        prev_stoch = prev_state["stoch"].flatten(-2, -1)
        concat_input = torch.cat((prev_stoch, prev_action), dim=-1)
        deter, para_stats = self.cell_layers(
            self.inp_layer(concat_input), prev_state, None, False)
        
        ims_stats = self.suff_stats_layer("ims", deter)
        stoch = ste_sample(self.get_dist(ims_stats))
        
        if return_stats:
            return deter, stoch, para_stats, ims_stats
        else:
            prior = {
                "stoch": stoch, "deter": deter,
                **ims_stats, **para_stats}
            return prior

    def suff_stats_layer(self, name, x):
        if name == "ims":
            x = self.ims_stat_layer(x)
        elif name == "obs":
            x = self.obs_stat_layer(x)
        else:
            raise NotImplementedError
        
        logit = x.unflatten(-1, (self.stoch, self.discrete))
        return {"logit": logit}
    
    def cell_layers(self, input, state, is_first, is_parallel):        
        if is_parallel:
            deter, is_first = swap(input), swap(is_first)
        else:
            deter, is_first = input, is_first
        
        stats = {}
        for id, layer in enumerate(self.rnn_layer):
            deter, cell_stats = layer(
                deter, is_first, state, is_parallel, id)
            stats.update(cell_stats)

        if is_parallel:
            deter = swap(deter)
            stats = {k: swap(v) for k, v in stats.items()}
        return torch.tanh(deter), stats

    def kl_loss(self, post, prior, free):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = kld(dist(post), dist(sg(prior)))
        dyn_loss = kld(dist(sg(post)), dist(prior))

        rep_loss = rep_loss.sum(dim=-1).mean()
        dyn_loss = dyn_loss.sum(dim=-1).mean()

        real_kl = dyn_loss
        ent = dist(post).entropy()
        ent = ent.sum(dim=-1).mean()

        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss, real_kl, ent


class JEPAWorldModel(nn.Module):
    """
    JEPA-based World Model for MuJoCo environments.

    Supports both proprioceptive (state-based) and visual observations.
    Uses continuous action encoding for MuJoCo's continuous action space.

    Key features:
    - CNN/MLP encoder with optional self-attention
    - RWKV predictor for temporal modeling
    - SIGReg for preventing feature collapse
    - Optional decoder for physical consistency (with weight decay)
    - MLP heads for reward and done prediction
    """

    def __init__(self,
                 video_log,
                 is_proprio,          # True for state-based, False for visual
                 obs_shape,
                 action_dim,          # Continuous action dimension
                 hidden,              # Embedding dimension (e.g., 512)
                 stem_ch,             # CNN stem channels (for visual)
                 min_res,             # Minimum resolution for CNN (for visual)
                 num_bin,             # Number of bins for reward prediction
                 max_bin,             # Max bin value for reward
                 gamma,               # Discount factor
                 lambd,               # Lambda for TD-lambda
                 tau,                 # EMA update rate
                 lr,                  # Learning rate
                 eps,                 # Adam epsilon
                 use_amp,             # Use automatic mixed precision
                 act,                 # Activation function
                 device,
                 # JEPA-specific parameters
                 use_self_attention=False,
                 self_attn_heads=4,
                 sigreg_weight=0.09,
                 sigreg_knots=17,
                 sigreg_num_proj=1024,
                 decoder_weight_init=1.0,
                 decoder_weight_min=0.0,
                 decoder_decay_steps=50000,
                 decoder_decay_type='linear',
                 rwkv_layers=4,
                 rwkv_heads=8,
                 rwkv_expand_factor=4,
                 ):
        super().__init__()

        # Basic attributes
        self.action_dim = action_dim
        self.hidden = hidden
        self.feat_dim = hidden  # JEPA: deterministic only
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.device = device
        self.batch_size = -1
        self.horizon = -1
        self.video_log = video_log
        self.is_proprio = is_proprio

        # AMP settings
        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = use_amp

        # JEPA-specific settings
        self.sigreg_weight = sigreg_weight
        self.use_self_attention = use_self_attention
        self.use_decoder = decoder_weight_init > 0

        # ==================== Encoder ====================
        if is_proprio:
            num_layer, encode_dim = 3, hidden * 2
            self.encoder = net.ProprioEncoder(obs_shape, encode_dim, num_layer, act)
        else:
            self.encoder = net.Encoder(obs_shape[0], obs_shape[-1], stem_ch, min_res, act)

        # Optional self-attention after encoder
        if use_self_attention:
            self.self_attn = TemporalSelfAttention(
                self.encoder.embed,
                num_heads=self_attn_heads,
                dropout=0.0,
                causal=False
            )
        else:
            self.self_attn = None

        # Projection to hidden dimension
        self.embed_proj = nn.Sequential(
            nn.Linear(self.encoder.embed, hidden),
            nn.LayerNorm(hidden)
        )

        # ==================== Action Encoder ====================
        # For continuous actions, use linear projection
        self.action_encoder = ContinuousActionEncoder(
            action_dim=action_dim,
            embed_dim=hidden,
            hidden_dim=hidden
        )

        # ==================== RWKV Predictor ====================
        # Note: action_dim is the hidden dim here since we use action_encoder
        self.predictor = RWKVPredictor(
            embed_dim=hidden,
            hidden_dim=hidden,
            action_dim=hidden,  # Using embedded action dim
            num_layers=rwkv_layers,
            num_heads=rwkv_heads,
            expand_factor=rwkv_expand_factor,
            dropout=0.0
        )

        # ==================== Decoder (optional) ====================
        if self.use_decoder:
            if is_proprio:
                num_layer, encode_dim = 3, hidden * 2
                self.decoder = net.ProprioDecoder(hidden, obs_shape, encode_dim, num_layer, act)
            else:
                self.decoder = net.Decoder(
                    hidden, self.encoder.out_ch, obs_shape[-1], stem_ch, min_res, act
                )
            self.decoder_scheduler = DecoderWeightScheduler(
                init_weight=decoder_weight_init,
                min_weight=decoder_weight_min,
                decay_type=decoder_decay_type,
                decay_steps=decoder_decay_steps
            )
        else:
            self.decoder = None
            self.decoder_scheduler = None

        # ==================== Prediction Heads ====================
        self.done_head = net.Head(hidden, 1, hidden, act)
        self.reward_head = net.Head(hidden, num_bin, hidden, act)

        # ==================== SIGReg ====================
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

        # ==================== Loss Functions ====================
        self.mse_loss = func.MseLoss(is_proprio)
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin)
        self.bce_logits_loss = F.binary_cross_entropy_with_logits
        self.pred_loss_fn = nn.MSELoss()

        # ==================== Optimizer ====================
        all_params = (
            params(self.encoder) +
            params(self.embed_proj) +
            params(self.action_encoder) +
            params(self.predictor) +
            params(self.done_head) +
            params(self.reward_head) +
            params(self.sigreg)
        )
        if self.self_attn is not None:
            all_params += params(self.self_attn)
        if self.decoder is not None:
            all_params += params(self.decoder)

        self.optimizer = torch.optim.AdamW(all_params, lr=lr, eps=eps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def preprocess(self, obs):
        """Preprocess observations to tensor."""
        if self.is_proprio:
            tensor_obs = torch.tensor(obs, dtype=self.tensor_dtype, device=self.device)
        else:
            tensor_obs = torch.tensor(obs, dtype=self.tensor_dtype, device=self.device) / 255
            tensor_obs = tensor_obs.permute(0, 3, 1, 2)[:, None]  # [B, 1, C, H, W]
        return tensor_obs

    def encode(self, obs):
        """
        Encode observations to deterministic embeddings.

        Args:
            obs: Observations (B, T, ...) - shape depends on is_proprio

        Returns:
            z: Embeddings (B, T, D)
        """
        # Encoder
        embed = self.encoder(obs)  # (B, T, encoder.embed)

        # Optional self-attention
        if self.self_attn is not None:
            embed = self.self_attn(embed)

        # Project to hidden dimension
        z = self.embed_proj(embed)  # (B, T, hidden)
        return z

    @torch.no_grad()
    def get_inference_feat(self, state, obs, is_first):
        """
        Get features for inference (environment interaction).

        Args:
            state: Previous state dict
            obs: Current observation (numpy array)
            is_first: Episode start flags

        Returns:
            feat: Features for agent (B, D)
            state: Updated state dict
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # Encode observation
            z = self.encode(self.preprocess(obs)).squeeze(1)  # (B, D)

            # Handle episode resets
            is_first = torch.tensor(is_first, dtype=self.tensor_dtype, device=self.device)
            if is_first.sum() > 0:
                init_state = self.initial(z.shape[0])
                for key, val in state.items():
                    if key == 'rwkv_state':
                        continue
                    num_axis = val.dim() - is_first.dim()
                    weight = is_first.unflatten(-1, [-1] + [1 for _ in range(num_axis)])
                    state[key] = val * (1 - weight) + init_state[key] * weight

            state['z'] = z

        return z, state

    @torch.no_grad()
    def update_inference_state(self, state, action):
        """
        Update state after taking an action (for inference).

        Args:
            state: Current state dict
            action: Action taken (continuous, shape (B, action_dim))

        Returns:
            state: Updated state dict
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            z = state['z']
            rwkv_state = state.get('rwkv_state', None)

            # Encode continuous action
            action_tensor = torch.tensor(action, dtype=self.tensor_dtype, device=self.device)
            action_emb = self.action_encoder(action_tensor)

            # Predict next embedding
            z_next, rwkv_state = self.predictor.step(z, action_emb, rwkv_state if rwkv_state else {})

            state['z'] = z_next
            state['rwkv_state'] = rwkv_state

        return state

    def initial(self, batch_size):
        """Initialize state for inference."""
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            z = torch.zeros(batch_size, self.hidden, device=self.device, dtype=self.tensor_dtype)
            return {'z': z, 'rwkv_state': {}}

    def init_imagine_buffer(self, batch_size, horizon):
        """Initialize buffers for imagination."""
        if self.batch_size != batch_size or self.horizon != horizon:
            init_zeros = lambda s: torch.zeros(s, dtype=self.tensor_dtype, device=self.device)
            self.batch_size, self.horizon = batch_size, horizon

            feat_size = (batch_size, horizon + 1, self.hidden)
            action_size = (batch_size, horizon, self.action_dim)

            self.feat_buffer = init_zeros(feat_size)
            self.action_buffer = init_zeros(action_size)

    @torch.no_grad()
    def get_video_frame(self, z, index):
        """Generate video frame from latent (if decoder exists and not proprio)."""
        if self.decoder is not None and not self.is_proprio:
            pred_frame = self.decoder(z[index, None])
            return pred_frame
        return None

    @torch.no_grad()
    def imagine_data(self, agent, obs, action, reward, done, is_first, horizon, logger=None, step=None):
        """
        Generate imagined trajectories for actor-critic training.

        Args:
            agent: Actor-critic agent
            obs: Observations
            action: Actions (B, T, action_dim)
            reward: Rewards
            done: Done flags
            is_first: Episode start flags
            horizon: Imagination horizon
            logger: Optional logger
            step: Current training step

        Returns:
            feat: Imagined features (B*T, H+1, D)
            action_buffer: Imagined actions (B*T, H, action_dim)
            discount: Discount factors
            reward: Predicted rewards
            weight: Importance weights
        """
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # Encode all observations
            z = self.encode(obs)  # (B, T, D)

            # Encode actions
            action_emb = self.action_encoder(action)  # (B, T, hidden)

            # Run RWKV forward to get states for each timestep
            _, rwkv_states = self.predictor(z, action_emb, is_first, state=None, parallel=True)

            # Flatten batch and time for imagination
            img_z = z.flatten(0, 1)  # (B*T, D)
            batch_size = img_z.shape[0]
            self.init_imagine_buffer(batch_size, horizon)

            video_index, pred_video = torch.randint(batch_size, (1,), device=self.device), []

            # Initialize RWKV state for imagination
            img_rwkv_state = {}

            for t in range(horizon):
                if logger is not None and not self.is_proprio and self.decoder is not None:
                    if step % self.video_log == 0:
                        frame = self.get_video_frame(img_z.unsqueeze(1), video_index)
                        if frame is not None:
                            pred_video.append(frame)

                self.feat_buffer[:, t] = img_z

                # Agent samples continuous action based on latent
                self.action_buffer[:, t] = agent.sample(img_z)

                # Encode action and predict next state
                action_emb = self.action_encoder(self.action_buffer[:, t])
                img_z, img_rwkv_state = self.predictor.step(img_z, action_emb, img_rwkv_state)

            self.feat_buffer[:, -1] = img_z

            # Predict rewards and dones from features
            discount = (self.done_head(self.feat_buffer[:, 1:]) < 0) * self.gamma
            reward = self.twohot_loss.decode(self.reward_head(self.feat_buffer[:, 1:]))
            weight = torch.cat((torch.ones_like(reward[:, :1]), discount[:, :-1]), dim=1)

        if logger is not None and len(pred_video) > 0 and not self.is_proprio:
            if step % self.video_log == 0:
                logger.log_video("Video/Imagination", torch.cat(pred_video, dim=1), step)

        return self.feat_buffer, self.action_buffer, discount, reward, weight

    def update(self, agent, obs, action, reward, done, is_first, logger=None, step=None):
        """
        Update world model parameters.

        JEPA loss = pred_loss + sigreg_weight * sigreg_loss + reward_loss + done_loss + decoder_weight * recon_loss

        Args:
            agent: Actor-critic agent (not used in world model update)
            obs: Observations
            action: Actions (B, T, action_dim)
            reward: Rewards (B, T, 1)
            done: Done flags (B, T, 1)
            is_first: Episode start flags (B, T)
            logger: Optional logger
            step: Current training step
        """
        self.train()

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # ==================== Encode ====================
            z = self.encode(obs)  # (B, T, D)

            # ==================== Encode Actions ====================
            action_emb = self.action_encoder(action)  # (B, T, hidden)

            # ==================== RWKV Predict ====================
            z_pred, _ = self.predictor(
                z[:, :-1], action_emb[:, :-1], is_first[:, :-1], parallel=True
            )  # (B, T-1, D)

            # Target: next timestep embeddings (stop gradient for JEPA)
            z_target = z[:, 1:].detach()  # (B, T-1, D)

            # ==================== Losses ====================
            # 1. Prediction loss (L2 in feature space)
            pred_loss = self.pred_loss_fn(z_pred, z_target)

            # 2. SIGReg regularization
            sigreg_loss = self.sigreg(z.transpose(0, 1))  # (T, B, D) format

            # 3. Reward and done prediction (from predicted embeddings)
            reward_hat = self.reward_head(z_pred)
            done_hat = self.done_head(z_pred)

            reward_loss = self.twohot_loss(reward_hat, reward[:, 1:])
            done_loss = self.bce_logits_loss(done_hat, done[:, 1:])

            # 4. Optional reconstruction loss (with decay)
            recon_loss = torch.tensor(0.0, device=self.device)
            decoder_weight = 0.0
            if self.decoder is not None and step is not None:
                decoder_weight = self.decoder_scheduler.get_weight(step)
                if decoder_weight > 0:
                    obs_hat = self.decoder(z)
                    recon_loss = self.mse_loss(obs_hat, obs)

            # ==================== Total Loss ====================
            total_loss = (
                pred_loss +
                self.sigreg_weight * sigreg_loss +
                reward_loss +
                done_loss +
                decoder_weight * recon_loss
            )

        # ==================== Backward ====================
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # ==================== Logging ====================
        if logger is not None:
            logger.log("WorldModel/pred_loss", pred_loss.item(), step)
            logger.log("WorldModel/sigreg_loss", sigreg_loss.item(), step)
            logger.log("WorldModel/reward_loss", reward_loss.item(), step)
            logger.log("WorldModel/done_loss", done_loss.item(), step)
            logger.log("WorldModel/decoder_weight", decoder_weight, step)

            if recon_loss.item() > 0:
                logger.log("WorldModel/recon_loss", recon_loss.item(), step)

            if step % self.video_log == 0 and not self.is_proprio and self.decoder is not None and decoder_weight > 0:
                video_index = torch.randint(obs.shape[0], (1,), device=self.device)
                logger.log_video("Video/Observation", obs[video_index], step)
                logger.log_video("Video/Reconstruction", obs_hat[video_index], step)
