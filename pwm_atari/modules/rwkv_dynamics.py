import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import OneHotCategorical

import modules.networks as net
from modules.rwkv6.official_wrappers import RWKV6OfficialCore, RWKV6StateManager
from modules.rwkv7.official_wrappers import RWKV7OfficialCore, RWKV7StateManager


ste_sample = lambda d: d.probs + (d.sample() - d.probs).detach()


RWKV_ARCH_REGISTRY = {
    "rwkv6": (RWKV6OfficialCore, RWKV6StateManager),
    "rwkv7": (RWKV7OfficialCore, RWKV7StateManager),
}


class RWKVDynamics(nn.Module):
    def __init__(
        self,
        stoch,
        hidden,
        discrete,
        action_dim,
        embed,
        act,
        device,
        unimix_ratio=0.01,
        n_layer=4,
        head_size=64,
        w0_bias=3.5,
        rwkv_arch="rwkv7",
        rwkv_kernel=False,
    ):
        super().__init__()
        self.stoch = stoch
        self.hidden = hidden
        self.discrete = discrete
        self.action_dim = action_dim
        self.unimix_ratio = unimix_ratio
        self.embed = embed
        self.act = act
        self.device = device

        self.stoch_dim = stoch * discrete
        self.n_embd = hidden
        self.head_size = head_size # default 64
        self.w0_bias = w0_bias
        self.rwkv_arch = str(rwkv_arch).lower()
        assert self.n_embd % self.head_size == 0, "hidden must be divisible by head_size"
        self.n_head = self.n_embd // self.head_size
        self.n_layer = 2  # default n_layer 2
        if self.rwkv_arch not in RWKV_ARCH_REGISTRY:
            raise ValueError(
                f"Unsupported RWKV architecture: {rwkv_arch}. "
                f"Expected one of {sorted(RWKV_ARCH_REGISTRY)}"
            )

        inp_dim = self.stoch_dim + action_dim
        self.inp_layer = net.InpLayer(inp_dim, self.n_embd, self.n_embd, act)
        self.ims_stat_layer = net.ImsStatLayer(self.n_embd, self.stoch_dim, act) # 想象统计层，从确定性状态预测随机状态的统计量
        self.obs_stat_layer = net.ObsStatLayer(embed, self.stoch_dim, act) # 观测统计层，从嵌入向量预测随机状态的统计量
        self.one_hot = lambda x: F.one_hot(x.long(), action_dim).to(x.dtype) # 将离散动作转换为 one-hot 向量

        core_cls, state_mgr_cls = RWKV_ARCH_REGISTRY[self.rwkv_arch]
        self.core = core_cls(self.n_embd, self.n_layer, self.head_size, w0_bias=self.w0_bias)
        self.state_mgr = state_mgr_cls(self.n_layer, self.n_embd, self.n_head, self.head_size)
        self.rwkv_kernel = rwkv_kernel

        self.register_buffer("init_deter", torch.zeros(1, self.hidden), persistent=False)

    def initial(self, batch_size):
        init = self.state_mgr.initial_state(batch_size, self.device, None)
        init_logit, init_stoch = self.get_init_stoch(self.init_deter.expand(batch_size, -1))
        init.update({"logit": init_logit, "stoch": init_stoch, "deter": self.init_deter.expand(batch_size, -1)})
        return init

    def get_init_stoch(self, deter):
        stats = self.suff_stats_layer("ims", deter)
        dist = self.get_dist(stats)
        return stats["logit"], dist.mode

    def get_deter(self, state):
        return state["deter"]

    def get_feat(self, state): # 特征向量输出=DETERMINISTIC隐状态+STOCHASTIC隐状态
        stoch = state["stoch"].flatten(-2, -1)
        return torch.cat((state["deter"], stoch), dim=-1)

    def get_flatten_stoch(self, state):
        return state["stoch"].flatten(-2, -1)

    def get_dist(self, state):
        probs = F.softmax(state["logit"], dim=-1)
        probs = probs * (1 - self.unimix_ratio) + self.unimix_ratio / self.discrete
        return OneHotCategorical(probs=probs)

    def suff_stats_layer(self, name, x):
        if name == "ims":
            x = self.ims_stat_layer(x)
        elif name == "obs":
            x = self.obs_stat_layer(x)
        else:
            raise NotImplementedError
        logit = x.unflatten(-1, (self.stoch, self.discrete))
        return {"logit": logit}

    def _step(self, x, state):
        return self.core.rnn_step(x, state)

    def _rnn_forward(self, latent, init, is_first):
        # RNN path for imagination (step-wise), using the selected RWKV formulas.
        batch_size, T = latent.shape[0], latent.shape[1]
        deter_list = []
        state = {k: v for k, v in init.items()}
        rwkv_cache = self.state_mgr.init_cache(batch_size, T, latent.device, latent.dtype)

        for t in range(T):
            mask = is_first[:, t] if is_first is not None else None
            state = self.state_mgr.reset(state, init, mask)
            x = latent[:, t]
            deter, state = self._step(x, state)
            deter_list.append(deter)
            for i in range(self.n_layer):
                rwkv_cache[f"rwkv_x_{i}"][:, t] = state[f"rwkv_x_{i}"]
                rwkv_cache[f"rwkv_s_{i}"][:, t] = state[f"rwkv_s_{i}"]
                rwkv_cache[f"rwkv_c_{i}"][:, t] = state[f"rwkv_c_{i}"]

        deter = torch.stack(deter_list, dim=1)
        return deter, rwkv_cache

    def _kernel_forward_with_cache(self, latent, is_first):
        # Kernel path for training (parallel_observe), owns per-timestep RWKV state caches.
        if not self.rwkv_kernel:
            raise RuntimeError("RWKV kernel path requested but rwkv_kernel is disabled.")
        if not latent.is_cuda:
            raise RuntimeError("RWKV kernel requires CUDA tensors.")

        batch_size, T = latent.shape[0], latent.shape[1]
        device = latent.device

        if is_first is None:
            reset_mask = torch.zeros(batch_size, T, device=device, dtype=torch.uint8)
        else:
            if is_first.dim() == 3 and is_first.shape[-1] == 1:
                is_first = is_first.squeeze(-1)
            reset_mask = (is_first > 0.5).to(torch.uint8)
        reset_mask[:, 0] = 1

        init = self.state_mgr.initial_state(batch_size, device, latent.dtype)
        s0_list = [init[f"rwkv_s_{i}"] for i in range(self.n_layer)]
        deter, rwkv_cache = self.core.forward_kernel_with_state(latent, s0_list, reset_mask)
        return deter, rwkv_cache

    def parallel_observe(self, embed, action, is_first):
        batch_size, T = action.shape[0], action.shape[1]
        init = self.initial(batch_size)
        obs_stats = self.suff_stats_layer("obs", embed) # logits
        oracle_stoch = ste_sample(self.get_dist(obs_stats)) # 采样观测随机状态

        if is_first is not None: # 强制第一个时间步为 episode 开始
            is_first = is_first.clone()
            is_first[:, 0] = 1

        flatten_stoch = oracle_stoch.flatten(-2, -1)
        concat_input = torch.cat((flatten_stoch, self.one_hot(action)), dim=-1)
        latent = self.inp_layer(concat_input)

        if self.rwkv_kernel:
            deter, rwkv_cache = self._kernel_forward_with_cache(latent, is_first)
        else:
            deter, rwkv_cache = self._rnn_forward(latent, init, is_first)
        ims_stats = self.suff_stats_layer("ims", deter[:, :-1])
        ims_stoch = ste_sample(self.get_dist(ims_stats))

        obs_stats = {k: v[:, 1:] for k, v in obs_stats.items()}
        obs_stoch = oracle_stoch[:, 1:]

        stats = {"deter": deter, **rwkv_cache}
        stats = {k: v[:, :-1] for k, v in stats.items()}
        post = {"stoch": obs_stoch, **obs_stats, **stats}
        prior = {"stoch": ims_stoch, **ims_stats, **stats}
        return post, prior, flatten_stoch, deter

    def img_step(self, prev_state, prev_action, return_stats=False):
        prev_stoch = prev_state["stoch"].flatten(-2, -1)
        concat_input = torch.cat((prev_stoch, self.one_hot(prev_action)), dim=-1)
        latent = self.inp_layer(concat_input)

        # 步骤1：预测确定性状态
        deter, state = self._step(latent, prev_state)
        # 步骤2：从确定性状态预测随机状态
        ims_stats = self.suff_stats_layer("ims", deter)
        stoch = ste_sample(self.get_dist(ims_stats))

        if return_stats:
            para_stats = {k: v for k, v in state.items() if k.startswith("rwkv_")}
            return deter, stoch, para_stats, ims_stats
        else:
            prior = {"stoch": stoch, "deter": deter, **ims_stats, **{k: v for k, v in state.items() if k.startswith("rwkv_")}}
            return prior

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
