import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.functions_losses as func
import modules.parallel_rnns as rnn
import modules.networks as net


params = lambda x: list(x.parameters())
permute = lambda x: x.permute(0, 3, 1, 2)
swap = lambda x: torch.transpose(x, 0, 1) 
to_param = lambda x: nn.Parameter(x)

def sigreg_loss(z):
    # z: (B*L, latent_dim)
    # 减去均值
    z_centered = z - z.mean(dim=0)
    # 方差约束：迫使每个维度的方差接近 1 (避免坍缩为一个点)
    std = torch.sqrt(z_centered.var(dim=0) + 1e-4)
    std_loss = torch.mean(torch.relu(1 - std))
    return std_loss


class ParallelWorldModel(nn.Module):
    def __init__(self,
                 video_log,
                 obs_shape,
                 num_action,
                 stoch,      # 保留用于兼容 yaml，不再用于离散概率采样
                 discrete,   # 同上
                 hidden,     # hidden 直接作为纯潜空间的 latent_dim
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
                 use_jacobian_reg=False,      # 兼容签名，由于采用确定性 JEPA 架构，已在反向传播中被废弃
                 jacobian_scale=0.0,          # 同上
                 jacobian_time_sample=4,      # 同上
                 jacobian_every=1,            # 同上
                 jacobian_probe_dist="rademacher", # 同上
                 jacobian_state_mode="all",   # 同上
                 rwkv_w0_bias=3.5,            # 同上
                 rwkv_arch="rwkv7",           # 同上
                ):
        super().__init__()
        self.num_action = num_action
        self.hidden = hidden
        self.feat_dim = hidden  # 特征维度变更为纯潜空间维度
        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.val_scale = val_scale
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

        # Encoder 已经升级为 CNN+Attention，输出 self.hidden 维度的 z
        self.encoder = net.Encoder(obs_shape[0], obs_shape[-1], stem_ch, min_res, act, latent_dim=self.hidden)
        # Decoder 接收自适应维度 self.hidden
        self.decoder = net.Decoder(self.hidden, self.encoder.out_ch, obs_shape[-1], stem_ch, min_res, act)
        
        # 使用确定性动力学算子 PSSM 替代原随机 RWKVDynamics
        self.dynamic = PSSM(
            hidden,
            num_action,
            act,
            device
        )
        
        self.done_head = net.Head(hidden, 1, hidden, act)
        self.reward_head = net.Head(hidden, num_bin, hidden, act)

        self.mse_loss = func.MseLoss() 
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin) 
        self.bce_logits_loss = F.binary_cross_entropy_with_logits 

        model_params = params(self.dynamic) + params(self.done_head) + params(self.reward_head)
        vae_params = params(self.encoder) + params(self.decoder)

        self.optimizer = torch.optim.AdamW(model_params + vae_params, lr=lr, eps=eps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) 

    @torch.no_grad()
    def preprocess(self, obs):
        tensor_obs = torch.tensor(
            obs, dtype=self.tensor_dtype, device=self.device) / 255
        tensor_obs = tensor_obs.permute(0, 3, 1, 2)[:, None] # [B, 1, C, H, W]
        return tensor_obs

    @torch.no_grad()
    def get_inference_feat(self, state, obs, is_first):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            # 获取真实特征 z
            true_z = self.encoder(self.preprocess(obs)).squeeze(1) 

            is_first_t = torch.tensor(is_first, dtype=self.tensor_dtype, device=self.device)
            if is_first_t.sum() > 0:
                init_state = self.initial(true_z.shape[0])
                mask = (is_first_t > 0.5).to(torch.bool).view(-1)
                
                for key, val in state.items():
                    cond = mask.bool().view(-1, *([1] * (val.dim() - 1)))
                    state[key] = torch.where(cond, init_state[key], val)

            state["z"] = true_z
            feat = true_z
        return feat, state
    
    @torch.no_grad()
    def update_inference_state(self, state, action): 
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            z_hat, next_state = self.dynamic.img_step(state["z"], action, state)
            next_state["z"] = z_hat
        return next_state

    def initial(self, batch_size):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            return self.dynamic.initial(batch_size)
    
    def init_imagine_buffer(self, batch_size, horizon):
        if self.batch_size != batch_size or self.horizon != horizon:
            init_zeros = lambda s: torch.zeros(s, dtype=self.tensor_dtype, device=self.device)
            self.batch_size, self.horizon = batch_size, horizon

            z_size = (batch_size, horizon+1, self.hidden)
            action_size = (batch_size, horizon) # Atari 的动作是离散的索引
            self.z_buffer = init_zeros(z_size)
            self.action_buffer = init_zeros(action_size)
    
    @torch.no_grad()
    def get_video_frame(self, z_sequence, index):
        pred_frame = self.decoder(z_sequence[index, None])
        return pred_frame

    @torch.no_grad() 
    def imagine_data(self, agent, obs, action, reward, done, is_first, horizon, logger=None, step=None):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            true_z = self.encoder(obs)
            z_full_pred, _, _, para_stats = self.dynamic.parallel_observe(true_z, action, is_first)
            
            # 提取最后一步状态，作为梦境推演的起点
            curr_z = true_z[:, -1]
            state = {k: v[:, -1] for k, v in para_stats.items()}
            
            batch_size = curr_z.shape[0]
            self.init_imagine_buffer(batch_size, horizon)

            video_index, pred_video = torch.randint(batch_size, (1,), device=self.device), []
            
            self.z_buffer[:, 0] = curr_z

            for t in range(horizon): 
                if logger is not None:
                    if step % self.video_log == 0:
                        pred_video += [self.get_video_frame(self.z_buffer[:, :t+1], video_index)[:, -1:]]
    
                self.action_buffer[:, t] = agent.sample(curr_z)
                curr_z, state = self.dynamic.img_step(curr_z, self.action_buffer[:, t], state)
                self.z_buffer[:, t+1] = curr_z

            feat = self.z_buffer # [B, T+1, hidden]
            discount = (self.done_head(self.z_buffer[:, 1:]) < 0) * self.gamma
            reward = self.twohot_loss.decode(self.reward_head(self.z_buffer[:, 1:]))
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

            # 1. 编码获取真实潜状态 z (带梯度，更新 Encoder)
            true_z = self.encoder(obs) 
            
            # 2. RWKV 确定性推演，获取预测状态 z_hat
            z_full_pred, z_hat, target_z, _ = self.dynamic.parallel_observe(true_z, action, is_first)
            
            # --- 损失 1：动力学对齐 (Cosine Distance) ---
            dyn_loss = 1 - F.cosine_similarity(z_hat, target_z.detach(), dim=-1).mean()
            
            # --- 损失 2：SIGReg 防坍缩 ---
            reg_loss = sigreg_loss(true_z.flatten(0, 1))
            
            # --- 损失 3：任务预测 (Reward & Done) ---
            done_hat = self.done_head(z_hat)
            reward_hat = self.reward_head(z_hat)
            done_loss = self.bce_logits_loss(done_hat, done[:, 1:])
            reward_loss = self.twohot_loss(reward_hat, reward[:, 1:])
            task_loss = done_loss + reward_loss
            
            # --- 损失 4：退火式图像重建 ---
            total_anneal_steps = 200000.0
            lambda_recon = 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / total_anneal_steps)))
            
            recon_loss = torch.tensor(0.0, device=self.device)
            if lambda_recon > 1e-3: 
                obs_hat = self.decoder(z_hat)
                recon_loss = self.mse_loss(obs_hat, obs[:, 1:])
            
            # 总损失聚合
            total_loss = self.dyn_scale * dyn_loss + self.rep_scale * reg_loss + lambda_recon * recon_loss + task_loss

        self.scaler.scale(total_loss).backward() 
        self.scaler.unscale_(self.optimizer) 
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0) 
        self.scaler.step(self.optimizer) 
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/dyn_loss", dyn_loss.item(), step)
            logger.log("WorldModel/reg_loss", reg_loss.item(), step)
            logger.log("WorldModel/reward_loss", reward_loss.item(), step)
            logger.log("WorldModel/done_loss", done_loss.item(), step)
            logger.log("WorldModel/lambda_recon", lambda_recon, step)
            if lambda_recon > 1e-3:
                logger.log("WorldModel/recon_loss", recon_loss.item(), step)

            if step % self.video_log == 0 and lambda_recon > 1e-3:
                video_index = torch.randint(obs.shape[0], (1,), device=self.device)
                logger.log_video("Video/Observation", obs[video_index, 1:], step)
                logger.log_video("Video/Reconstruction", obs_hat[video_index], step)


class PSSM(nn.Module):
    """
    确定性的 RWKV 动力学算子 (适用于 Atari 的离散动作空间)
    取代了原版基于概率分布采样的架构
    """
    def __init__(self, hidden, num_action, act, device):
        super().__init__()
        self.hidden = hidden
        self.num_action = num_action
        self.act = act
        self.device = device
        self.num_rnns = 2

        inp_dim = hidden + num_action

        self.rnn_layer = self.init_cell()
        self.inp_layer = net.InpLayer(inp_dim, hidden, hidden, act)
        # 离散动作转化为独热编码以兼容网络输入
        self.one_hot = lambda x: F.one_hot(x.long(), num_action).float()
        
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
        init = {k: v.expand(batch_size, v.shape[-1]) for k, v in self.cell_ws.items()}
        init["deter"] = self.init_deter.expand(batch_size, -1)
        return init
    
    def parallel_observe(self, embed_z, action, is_first):
        # embed_z: (B, L, hidden)
        init = self.initial(action.shape[0])
        
        # 将离散动作转为 one-hot 并拼接到状态上
        concat_input = torch.cat((embed_z, self.one_hot(action)), dim=-1) 
        latent, mask = self.inp_layer(concat_input), is_first
        
        # 经过并行 RWKV 单元，输出完整的预测时序序列 z_full_pred
        z_full_pred, para_stats = self.cell_layers(latent, init, mask, True)
        
        # 错位对齐用于计算 Dyn Loss：用 z_{0:t-1} 去预测 z_{1:t}
        pred_z = z_full_pred[:, :-1]
        target_z = embed_z[:, 1:]
        
        return z_full_pred, pred_z, target_z, para_stats

    def img_step(self, prev_z, prev_action, state):
        # 单步推演 (用于 imagine_data 梦境阶段)
        concat_input = torch.cat((prev_z, self.one_hot(prev_action)), dim=-1)
        latent = self.inp_layer(concat_input)
        
        z_hat, next_state = self.cell_layers(latent, state, None, False)
        next_state["deter"] = z_hat 
        
        return z_hat, next_state
    
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