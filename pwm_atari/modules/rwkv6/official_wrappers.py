import importlib
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

_KERNEL_HEAD_SIZE = None


def _load_official_blocks(head_size):
    global _KERNEL_HEAD_SIZE
    if _KERNEL_HEAD_SIZE is None:
        _KERNEL_HEAD_SIZE = head_size
    elif _KERNEL_HEAD_SIZE != head_size:
        raise RuntimeError(
            f"RWKV6 kernel already initialized with head_size={_KERNEL_HEAD_SIZE}, "
            f"requested head_size={head_size}."
        )
    os.environ.setdefault("RWKV_MY_TESTING", "")
    if "x060" not in os.environ["RWKV_MY_TESTING"]:
        os.environ["RWKV_MY_TESTING"] = os.environ["RWKV_MY_TESTING"] + "x060"
    os.environ["RWKV_HEAD_SIZE"] = str(head_size)
    mod = importlib.import_module("modules.rwkv6.official_blocks")
    return importlib.reload(mod)


def _view_1d(x):
    return x.view(-1) if x.dim() > 1 else x


def time_mixing_rnn(
    H,
    N,
    x,
    x_prev,
    state,
    time_maa_x,
    time_maa_w,
    time_maa_k,
    time_maa_v,
    time_maa_r,
    time_maa_g,
    time_maa_w1,
    time_maa_w2,
    time_decay,
    time_decay_w1,
    time_decay_w2,
    time_faaaa,
    receptance,
    key,
    value,
    gate,
    output,
    ln_w,
    ln_b,
):
    time_maa_x = _view_1d(time_maa_x)
    time_maa_w = _view_1d(time_maa_w)
    time_maa_k = _view_1d(time_maa_k)
    time_maa_v = _view_1d(time_maa_v)
    time_maa_r = _view_1d(time_maa_r)
    time_maa_g = _view_1d(time_maa_g)
    time_decay = _view_1d(time_decay)

    xx = x_prev - x
    xxx = x + xx * time_maa_x
    xxx = torch.tanh(xxx @ time_maa_w1).view(x.shape[0], 5, -1).permute(1, 0, 2)
    xxx = torch.bmm(xxx, time_maa_w2).permute(1, 0, 2)
    mw, mk, mv, mr, mg = xxx.unbind(dim=1)

    xw = x + xx * (time_maa_w + mw)
    xk = x + xx * (time_maa_k + mk)
    xv = x + xx * (time_maa_v + mv)
    xr = x + xx * (time_maa_r + mr)
    xg = x + xx * (time_maa_g + mg)

    r = F.linear(xr, receptance)
    k = F.linear(xk, key)
    v = F.linear(xv, value)
    g = F.silu(F.linear(xg, gate))
    ww = torch.tanh(xw @ time_decay_w1) @ time_decay_w2
    w = torch.exp(-torch.exp(time_decay + ww).view(-1, H, N))

    r_h = r.float().view(-1, H, N)
    k_h = k.float().view(-1, H, N)
    v_h = v.float().view(-1, H, N)
    u = time_faaaa.float().view(1, H, 1, N)
    kv = v_h.unsqueeze(-1) * k_h.unsqueeze(-2)
    out = (((kv * u) + state.float()) * r_h.unsqueeze(-2)).sum(dim=-1).reshape(x.shape[0], -1)
    state = state.float() * w.unsqueeze(-2) + kv

    out = out.to(ln_w.dtype)
    # Match the official x060 block path so training (parallel) and imagination (RNN step)
    # use the same normalization numerics.
    out = F.group_norm(out, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)
    out = F.linear(out * g, output)
    return out, x, state


def channel_mixing_rnn(x, x_prev, time_maa_k, time_maa_r, key, receptance, value):
    time_maa_k = _view_1d(time_maa_k)
    time_maa_r = _view_1d(time_maa_r)
    xx = x_prev - x
    xk = x + xx * time_maa_k
    xr = x + xx * time_maa_r
    k = torch.relu(F.linear(xk, key)) ** 2
    kv = F.linear(k, value)
    return torch.sigmoid(F.linear(xr, receptance)) * kv, x


class RWKV6OfficialBlock(nn.Module):
    def __init__(self, args, layer_id, att_cls, ffn_cls):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = att_cls(args, layer_id)
        self.ffn = ffn_cls(args, layer_id)

    def rnn_step(self, x, x_prev, time_state, c_prev):
        xx = F.layer_norm(x, (self.args.n_embd,), weight=self.ln1.weight, bias=self.ln1.bias)
        xx, x_prev, time_state = time_mixing_rnn(
            self.att.n_head,
            self.att.head_size,
            xx,
            x_prev,
            time_state,
            self.att.time_maa_x,
            self.att.time_maa_w,
            self.att.time_maa_k,
            self.att.time_maa_v,
            self.att.time_maa_r,
            self.att.time_maa_g,
            self.att.time_maa_w1,
            self.att.time_maa_w2,
            self.att.time_decay,
            self.att.time_decay_w1,
            self.att.time_decay_w2,
            self.att.time_faaaa,
            self.att.receptance.weight,
            self.att.key.weight,
            self.att.value.weight,
            self.att.gate.weight,
            self.att.output.weight,
            self.att.ln_x.weight,
            self.att.ln_x.bias,
        )
        x = x + xx

        xx = F.layer_norm(x, (self.args.n_embd,), weight=self.ln2.weight, bias=self.ln2.bias)
        xx, c_prev = channel_mixing_rnn(
            xx,
            c_prev,
            self.ffn.time_maa_k,
            self.ffn.time_maa_r,
            self.ffn.key.weight,
            self.ffn.receptance.weight,
            self.ffn.value.weight,
        )
        x = x + xx
        return x, x_prev, time_state, c_prev

    def forward_kernel_with_state(self, x, s0, reset_mask):
        xx = self.ln1(x)
        x_in = xx
        xx, s_last, s_t = self.att.forward_with_state(xx, s0, reset_mask, True)
        x = x + xx

        xx = self.ln2(x)
        c_in = xx
        xx = self.ffn.forward_with_reset(xx, reset_mask)
        x = x + xx
        return x, x_in, s_last, s_t, c_in


class RWKV6OfficialCore(nn.Module):
    def __init__(self, n_embd, n_layer, head_size, **_kwargs):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.head_size = head_size
        assert self.n_embd % self.head_size == 0, "n_embd must be divisible by head_size"
        self.n_head = self.n_embd // self.head_size

        args = types.SimpleNamespace(
            n_embd=n_embd,
            n_layer=n_layer,
            head_size=head_size,
            dim_att=n_embd,
            dim_ffn=n_embd * 4,
            my_testing="x060",
        )
        blocks_mod = _load_official_blocks(head_size)
        self._blocks_mod = blocks_mod
        att_cls = blocks_mod.RWKV_Tmix_x060
        ffn_cls = blocks_mod.RWKV_CMix_x060
        self.blocks = nn.ModuleList([RWKV6OfficialBlock(args, i, att_cls, ffn_cls) for i in range(n_layer)])

    def rnn_step(self, x_t, state):
        if x_t.dtype != state["rwkv_x_0"].dtype:
            raise RuntimeError("RWKV6 RNN dtype mismatch between input and state.")
        for i, block in enumerate(self.blocks):
            x_prev = state[f"rwkv_x_{i}"]
            time_state = state[f"rwkv_s_{i}"]
            c_prev = state[f"rwkv_c_{i}"]

            x_t, x_prev, time_state, c_prev = block.rnn_step(x_t, x_prev, time_state, c_prev)

            state[f"rwkv_x_{i}"] = x_prev
            state[f"rwkv_s_{i}"] = time_state
            state[f"rwkv_c_{i}"] = c_prev
        return x_t, state

    def forward_kernel_with_state(self, x, s0_list, reset_mask):
        if x.dtype != self.blocks[0].ln1.weight.dtype:
            raise RuntimeError("RWKV6 kernel input dtype mismatch with parameters.")
        rwkv_cache = {}
        for i, block in enumerate(self.blocks):
            x, x_in, s_last, s_t, c_in = block.forward_kernel_with_state(x, s0_list[i], reset_mask)
            rwkv_cache[f"rwkv_x_{i}"] = x_in
            rwkv_cache[f"rwkv_s_{i}"] = s_t
            rwkv_cache[f"rwkv_c_{i}"] = c_in
        return x, rwkv_cache


class RWKV6StateManager:
    def __init__(self, n_layer, n_embd, n_head, head_size):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = head_size

    def initial_state(self, batch_size, device, dtype):
        if dtype is None:
            dtype = torch.float32
        state = {}
        for i in range(self.n_layer):
            state[f"rwkv_x_{i}"] = torch.zeros(batch_size, self.n_embd, device=device, dtype=dtype)
            state[f"rwkv_s_{i}"] = torch.zeros(
                batch_size, self.n_head, self.head_size, self.head_size, device=device, dtype=torch.float32
            )
            state[f"rwkv_c_{i}"] = torch.zeros(batch_size, self.n_embd, device=device, dtype=dtype)
        return state

    def reset(self, state, init_state, is_first_mask):
        if is_first_mask is None:
            return state
        mask = (is_first_mask > 0.5).to(torch.bool).view(-1)
        if mask.sum() == 0:
            return state
        for k, v in state.items():
            if k in init_state:
                cond = mask.view(-1, *([1] * (v.dim() - 1)))
                state[k] = torch.where(cond, init_state[k], v)
        return state

    def init_cache(self, batch_size, T, device, dtype):
        cache = {}
        for i in range(self.n_layer):
            cache[f"rwkv_x_{i}"] = torch.empty(batch_size, T, self.n_embd, device=device, dtype=dtype)
            cache[f"rwkv_c_{i}"] = torch.empty(batch_size, T, self.n_embd, device=device, dtype=dtype)
            cache[f"rwkv_s_{i}"] = torch.empty(
                batch_size, T, self.n_head, self.head_size, self.head_size, device=device, dtype=torch.float32
            )
        return cache
