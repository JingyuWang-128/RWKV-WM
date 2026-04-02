import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import OneHotCategorical

import modules.functions_losses as func
import modules.parallel_rnns as rnn
from modules.rwkv_dynamics import RWKVDynamics
import modules.networks as net


params = lambda x: list(x.parameters())
permute = lambda x: x.permute(0, 3, 1, 2)
swap = lambda x: torch.transpose(x, 0, 1) 
ste_sample = lambda d: d.probs + (d.sample() - d.probs).detach()
to_param = lambda x: nn.Parameter(x)


class ParallelWorldModel(nn.Module):
    def __init__(self,
                 video_log,
                 obs_shape,
                 num_action,
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
                 rwkv_kernel=False,
                 use_jacobian_reg=False,
                 jacobian_scale=0.0,
                 jacobian_time_sample=4,
                 jacobian_every=1,
                 jacobian_probe_dist="rademacher",
                 jacobian_state_mode="all",
                 rwkv_w0_bias=3.5,
                 rwkv_arch="rwkv7",
                ):
        super().__init__()
        self.num_action = num_action
        self.hidden = hidden
        self.stoch_dim = stoch * discrete # 随机状态的总维度 = 随机状态数 × 离散类别数
        self.feat_dim = self.stoch_dim + hidden # 特征维度 = 随机状态总维度 + 确定性状态维度
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

        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = use_amp
        self.use_jacobian_reg = use_jacobian_reg
        self.jacobian_scale = jacobian_scale
        self.jacobian_time_sample = jacobian_time_sample
        self.jacobian_every = max(1, jacobian_every)
        self.jacobian_probe_dist = jacobian_probe_dist.lower()
        self.jacobian_state_mode = jacobian_state_mode.lower()
        self.rwkv_w0_bias = float(rwkv_w0_bias)
        self.rwkv_arch = str(rwkv_arch).lower()
        valid_jacobian_modes = {"all", "time_only", "s_only"}
        if self.jacobian_state_mode not in valid_jacobian_modes:
            raise ValueError(
                f"Unsupported Jacobian state mode: {jacobian_state_mode}. "
                f"Expected one of {sorted(valid_jacobian_modes)}"
            )

        self.encoder = net.Encoder(obs_shape[0], obs_shape[-1], stem_ch, min_res, act)
        self.decoder = net.Decoder(
            self.stoch_dim, self.encoder.out_ch, obs_shape[-1], stem_ch, min_res, act)
        
        self.dynamic = RWKVDynamics(
            stoch,
            hidden,
            discrete,
            num_action,
            self.encoder.embed,
            act,
            device,
            w0_bias=self.rwkv_w0_bias,
            rwkv_arch=self.rwkv_arch,
            rwkv_kernel=rwkv_kernel,
        )
        self.done_head = net.Head(hidden, 1, hidden, act)
        self.reward_head = net.Head(hidden, num_bin, hidden, act)

        self.mse_loss = func.MseLoss() # MSE损失用于图像重建
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin) # 对数两热编码损失用于奖励预测
        self.bce_logits_loss = F.binary_cross_entropy_with_logits # 二元交叉熵损失用于done预测

        model_params = params(self.dynamic) + params(self.done_head) + params(self.reward_head)
        vae_params = params(self.encoder) + params(self.decoder)

        self.optimizer = torch.optim.AdamW(model_params + vae_params, lr=lr, eps=eps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) # 用于混合精度训练

    @staticmethod
    def _flatten_state_dict(state):
        return {k: v.flatten(0, 1) for k, v in state.items()}

    def _sample_probe(self, h_next):
        if self.jacobian_probe_dist == "gaussian":
            probe = torch.randn_like(h_next)
            probe = probe / (torch.linalg.norm(probe, dim=-1, keepdim=True) + 1e-6)
            return probe
        if self.jacobian_probe_dist != "rademacher":
            raise ValueError(f"Unsupported Jacobian probe distribution: {self.jacobian_probe_dist}")
        probe = torch.empty_like(h_next).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        return probe / math.sqrt(h_next.shape[-1])

    def _select_jacobian_keys(self, post):
        all_keys = [k for k in post.keys() if k.startswith("rwkv_")]
        if self.jacobian_state_mode == "all":
            return all_keys
        if self.jacobian_state_mode == "time_only":
            return [k for k in all_keys if "_x_" in k or "_s_" in k]
        return [k for k in all_keys if "_s_" in k]

    def _compute_jacobian_loss(self, post, action):
        grad_keys = self._select_jacobian_keys(post)
        recurrent_state_keys = [k for k in post.keys() if k.startswith("rwkv_")]
        if not grad_keys or not recurrent_state_keys:
            zero = torch.zeros((), device=self.device, dtype=torch.float32)
            return zero, zero, zero

        post_len = post["deter"].shape[1]
        usable_steps = post_len - 1
        t_sample = min(self.jacobian_time_sample, usable_steps)
        if t_sample <= 0:
            zero = torch.zeros((), device=self.device, dtype=torch.float32)
            return zero, zero, zero

        max_start = usable_steps - t_sample
        start = random.randint(0, max_start) if max_start > 0 else 0
        stop = start + t_sample

        jac_state = {
            "stoch": post["stoch"][:, start:stop].detach().float(),
        }
        grad_inputs = []
        for key in recurrent_state_keys:
            value = post[key][:, start:stop].detach().float().clone()
            if key in grad_keys:
                value = value.requires_grad_(True)
                grad_inputs.append(value)
            jac_state[key] = value

        action_jac = action[:, 1 + start : 1 + stop].detach().float().reshape(-1)
        state_flat = self._flatten_state_dict(jac_state)

        with torch.autocast(device_type=self.device_type, enabled=False):
            next_state_flat = self.dynamic.img_step(state_flat, action_jac)
            h_next = next_state_flat["deter"].float()
            probe = self._sample_probe(h_next)
            jtv_parts = torch.autograd.grad(
                outputs=h_next,
                inputs=grad_inputs,
                grad_outputs=probe,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )

        per_sample_sq = torch.zeros(h_next.shape[0], device=h_next.device, dtype=torch.float32)
        for jtv in jtv_parts:
            jtv = jtv.float().flatten(0, 1).flatten(1)
            sq = jtv.pow(2).sum(dim=-1)
            per_sample_sq = per_sample_sq + sq
        raw_jacobian_loss = per_sample_sq.mean()
        jacobian_norm_est = per_sample_sq.sqrt().mean()
        return raw_jacobian_loss, jacobian_norm_est

    @torch.no_grad()
    def preprocess(self, obs):
        tensor_obs = torch.tensor(
            obs, dtype=self.tensor_dtype, device=self.device) / 255
        tensor_obs = tensor_obs.permute(0, 3, 1, 2)[:, None] # [B, 1, C, H, W]
        return tensor_obs

    @torch.no_grad()
    def get_inference_feat(self, state, obs, is_first):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            embed = self.encoder(self.preprocess(obs)).squeeze(1)
            obs_stats = self.dynamic.suff_stats_layer("obs", embed) # 计算观测的充分统计量--logits
            obs_stoch = ste_sample(self.dynamic.get_dist(obs_stats)) # 捕获环境的随机性

            is_first = torch.tensor(is_first, dtype=self.tensor_dtype, device=self.device)
            if is_first.sum() > 0:
                init_state = self.initial(obs_stoch.shape[0])
                mask = (is_first > 0.5).to(torch.bool).view(-1)
                
                for key, val in state.items():
                    cond = mask.bool().view(-1, *([1] * (val.dim() - 1)))
                    state[key] = torch.where(cond, init_state[key], val)

            state.update({"stoch": obs_stoch, **obs_stats}) # 更新状态字典
        return self.dynamic.get_feat(state), state
    
    @torch.no_grad()
    def update_inference_state(self, state, action): # 根据动作预测下一个状态
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            img_step_stats = self.dynamic.img_step(state, action, True)
            deter, _, para_stats, _ = img_step_stats
            state.update({"deter": deter, **para_stats}) # para_stats就是RWKV中保存的内部状态
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
            action_size = (batch_size, horizon)
            self.deter_buffer = init_zeros(deter_size)
            self.stoch_buffer = init_zeros(stoch_size)
            self.action_buffer = init_zeros(action_size)
    
    @torch.no_grad()
    def get_video_frame(self, prior, index):
        stoch = self.dynamic.get_flatten_stoch(prior)
        pred_frame = self.decoder(stoch[index, None])
        return pred_frame

    @torch.no_grad() # 生成想象轨迹数据，用于策略训练
    def imagine_data(self, agent, obs, action, reward, done, is_first, horizon, logger=None, step=None):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            state, _, _, _ = self.dynamic.parallel_observe(self.encoder(obs), action, is_first)
            img_state = {k: v.flatten(0, 1) for k, v in state.items()} # `[B, T]` → `[B*T]` 展平批次和时间维度
            batch_size = self.dynamic.get_feat(img_state).shape[0]
            self.init_imagine_buffer(batch_size, horizon)

            video_index, pred_video = torch.randint(batch_size, (1,), device=self.device), []

            for t in range(horizon): # 生成想象轨迹数据，长度为horizon
                if logger is not None:
                    if step % self.video_log == 0:
                        pred_video += [self.get_video_frame(img_state, video_index)]
    
                self.deter_buffer[:, t] = self.dynamic.get_deter(img_state)
                self.stoch_buffer[:, t] = self.dynamic.get_flatten_stoch(img_state)
                self.action_buffer[:, t] = agent.sample(
                    torch.cat((self.deter_buffer[:, t], self.stoch_buffer[:, t]), dim=-1)) # 智能体根据当前特征采样动作（整个state序列）
                img_state = self.dynamic.img_step(img_state, self.action_buffer[:, t]) # 根据动作预测下一个状态
            
            self.deter_buffer[:, -1] = self.dynamic.get_deter(img_state) # 存储最后一步的状态
            self.stoch_buffer[:, -1] = self.dynamic.get_flatten_stoch(img_state) # 存储最后一步的随机状态

            feat = torch.cat((self.deter_buffer, self.stoch_buffer), dim=-1) # [B, T+1, H+S]
            discount = (self.done_head(self.deter_buffer[:, 1:]) < 0) * self.gamma
            reward = self.twohot_loss.decode(self.reward_head(self.deter_buffer[:, 1:]))
            weight = torch.cumprod(
                torch.cat((torch.ones_like(reward[:, :1]), discount[:, :-1]), dim=1),
                dim=1,
            )
            
        if logger is not None:
            if step % self.video_log == 0:
                logger.log_video("Video/Imagination", torch.cat(pred_video, dim=1), step)

        return feat, self.action_buffer, discount, reward, weight

    def update(self, agent, obs, action, reward, done, is_first, logger=None, step=None):
        self.train()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):

            # post: 基于观测的后验状态；prior: 基于动态模型的先验状态；stoch: 后验的随机状态；deter: 确定性状态
            post, prior, stoch, deter = self.dynamic.parallel_observe(self.encoder(obs), action, is_first)
            # rep_loss: 表示损失；dyn_loss: 动态损失；real_kl: 真实的KL散度值；ent: 后验分布的熵
            dyn_loss, rep_loss, real_kl, ent = self.dynamic.kl_loss(post, prior, self.kl_free)

            obs_hat = self.decoder(stoch) # 重建观测
            done_hat = self.done_head(deter)
            reward_hat = self.reward_head(deter)

            recon_loss = self.mse_loss(obs_hat, obs)
            done_loss = self.bce_logits_loss(done_hat, done)
            reward_loss = self.twohot_loss(reward_hat, reward)
            head_loss = done_loss + reward_loss
            model_loss = self.dyn_scale * dyn_loss + head_loss
            vae_loss = recon_loss + self.rep_scale * rep_loss

        raw_jacobian_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        jacobian_norm = torch.zeros((), device=self.device, dtype=torch.float32)
        should_regularize = (
            self.use_jacobian_reg
            and self.jacobian_scale > 0.0
            and (step is None or step % self.jacobian_every == 0)
        )
        if should_regularize:
            raw_jacobian_loss, jacobian_norm = self._compute_jacobian_loss(post, action)

        total_loss = model_loss + vae_loss + self.jacobian_scale * raw_jacobian_loss
        self.scaler.scale(total_loss).backward() # 缩放损失并反向传播（混合精度）
        self.scaler.unscale_(self.optimizer) # 取消缩放梯度（用于梯度裁剪）
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0) # 梯度裁剪，防止梯度爆炸
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
            if should_regularize:
                logger.log("WorldModel/raw_jacobian_loss", raw_jacobian_loss.item(), step)
                logger.log("WorldModel/jacobian_norm_est", jacobian_norm.item(), step)

            if step % self.video_log == 0:
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
        self.one_hot = lambda x: F.one_hot(x.long(), action_dim).to(x.dtype)
        
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
        concat_input = torch.cat((flatten_stoch, self.one_hot(action)), dim=-1)
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
        concat_input = torch.cat((prev_stoch, self.one_hot(prev_action)), dim=-1)
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
        return deter, stats

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
