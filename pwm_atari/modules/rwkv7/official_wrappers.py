import os
import types
import importlib

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
            f"RWKV7 kernel already initialized with head_size={_KERNEL_HEAD_SIZE}, "
            f"requested head_size={head_size}."
        )
    os.environ.setdefault("RWKV_MY_TESTING", "")
    if "x070" not in os.environ["RWKV_MY_TESTING"]:
        os.environ["RWKV_MY_TESTING"] = os.environ["RWKV_MY_TESTING"] + "x070"
    os.environ["RWKV_HEAD_SIZE"] = str(head_size)
    mod = importlib.import_module("modules.rwkv7.official_blocks")
    return importlib.reload(mod)


def _view_1d(x):
    return x.view(-1) if x.dim() > 1 else x


def time_mixing_rnn(
    layer_id,
    H,
    N,
    x,
    x_prev,
    v_first,
    state,
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    w0,
    w1,
    w2,
    a0,
    a1,
    a2,
    v0,
    v1,
    v2,
    g1,
    g2,
    k_k,
    k_a,
    r_k,
    kw,
    vw,
    rw,
    ow,
    ln_w,
    ln_b,
):
    # Batched form of official RWKV7 RNN time_mixing__ from rwkv_v7_demo_rnn.py.
    x_r = _view_1d(x_r)
    x_w = _view_1d(x_w)
    x_k = _view_1d(x_k)
    x_v = _view_1d(x_v)
    x_a = _view_1d(x_a)
    x_g = _view_1d(x_g)

    w0 = _view_1d(w0)
    a0 = _view_1d(a0)
    v0 = _view_1d(v0)
    k_k = _view_1d(k_k)
    k_a = _view_1d(k_a)

    xx = x_prev - x
    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

    r = F.linear(xr, rw)
    w = torch.tanh(xw @ w1) @ w2
    k = F.linear(xk, kw)
    v = F.linear(xv, vw)
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = k * k_k
    kk = F.normalize(kk.view(-1, H, N), dim=-1, p=2.0).view(x.shape[0], -1)
    k = k * (1 + (a - 1) * k_a)

    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = w0 + w.float()
    w = torch.exp(-0.6065306597 / (1.0 + torch.exp(-w)))

    vk = v.view(-1, H, N, 1) @ k.view(-1, H, 1, N)
    ab = (-kk).view(-1, H, N, 1) @ (kk * a).view(-1, H, 1, N)
    state = state * w.view(-1, H, 1, N) + torch.matmul(state, ab.float()) + vk.float()

    state_f = state.float()
    out = torch.matmul(state_f, r.float().view(-1, H, N, 1)).view(x.shape[0], -1)
    out = out.to(ln_w.dtype)
    out = F.group_norm(out, num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5)
    r_view = r.view(-1, H, N)
    k_view = k.view(-1, H, N)
    out = out + (
        (r_view * k_view * r_k.view(1, H, N))
        .sum(dim=-1, keepdim=True)
        * v.view(-1, H, N)
    ).view(x.shape[0], -1)
    out = F.linear(out * g, ow)
    return out, x, state, v_first


def channel_mixing_rnn(x, x_prev, x_k, kw, vw):
    # Batched form of official RWKV7 RNN channel_mixing__ from rwkv_v7_demo_rnn.py.
    x_k = _view_1d(x_k)
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(F.linear(k, kw)) ** 2
    return F.linear(k, vw), x


class RWKV7OfficialBlock(nn.Module):
    def __init__(self, args, layer_id, att_cls, ffn_cls):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = att_cls(args, layer_id)
        self.ffn = ffn_cls(args, layer_id)

    def rnn_step(self, x, x_prev, time_state, c_prev, v_first):
        xx = F.layer_norm(x, (self.args.n_embd,), weight=self.ln1.weight, bias=self.ln1.bias)
        xx, x_prev, time_state, v_first = time_mixing_rnn(
            self.layer_id,
            self.att.n_head,
            self.att.head_size,
            xx,
            x_prev,
            v_first,
            time_state,
            self.att.x_r,
            self.att.x_w,
            self.att.x_k,
            self.att.x_v,
            self.att.x_a,
            self.att.x_g,
            self.att.w0,
            self.att.w1,
            self.att.w2,
            self.att.a0,
            self.att.a1,
            self.att.a2,
            self.att.v0,
            self.att.v1,
            self.att.v2,
            self.att.g1,
            self.att.g2,
            self.att.k_k,
            self.att.k_a,
            self.att.r_k,
            self.att.key.weight,
            self.att.value.weight,
            self.att.receptance.weight,
            self.att.output.weight,
            self.att.ln_x.weight,
            self.att.ln_x.bias,
        )
        x = x + xx

        xx = F.layer_norm(x, (self.args.n_embd,), weight=self.ln2.weight, bias=self.ln2.bias)
        xx, c_prev = channel_mixing_rnn(xx, c_prev, self.ffn.x_k, self.ffn.key.weight, self.ffn.value.weight)
        x = x + xx
        return x, x_prev, time_state, c_prev, v_first

    def forward_kernel_with_state(self, x, v_first, s0, reset_mask):
        xx = self.ln1(x)
        x_in = xx
        xx, v_first, s_last, s_t = self.att.forward_with_state(xx, v_first, s0, reset_mask, True)
        x = x + xx

        xx = self.ln2(x)
        c_in = xx
        xx = self.ffn.forward_with_reset(xx, reset_mask)
        x = x + xx
        return x, v_first, x_in, s_last, s_t, c_in


class RWKV7OfficialCore(nn.Module):
    def __init__(self, n_embd, n_layer, head_size, w0_bias=3.5):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.head_size = head_size
        self.w0_bias = w0_bias
        assert self.n_embd % self.head_size == 0, "n_embd must be divisible by head_size"
        self.n_head = self.n_embd // self.head_size

        args = types.SimpleNamespace(
            n_embd=n_embd,
            n_layer=n_layer,
            head_size=head_size,
            dim_att=n_embd,
            my_testing="x070",
            w0_bias=w0_bias,
        )
        blocks_mod = _load_official_blocks(head_size)
        self._blocks_mod = blocks_mod
        att_cls = blocks_mod.RWKV_Tmix_x070
        ffn_cls = blocks_mod.RWKV_CMix_x070
        self.blocks = nn.ModuleList(
            [RWKV7OfficialBlock(args, i, att_cls, ffn_cls) for i in range(n_layer)]
        )

    def rnn_step(self, x_t, state):
        if x_t.dtype != state["rwkv_x_0"].dtype:
            raise RuntimeError("RWKV7 RNN dtype mismatch between input and state.")
        v_first = None
        for i, block in enumerate(self.blocks):
            x_prev = state[f"rwkv_x_{i}"]
            time_state = state[f"rwkv_s_{i}"]
            c_prev = state[f"rwkv_c_{i}"]

            x_t, x_prev, time_state, c_prev, v_first = block.rnn_step(
                x_t, x_prev, time_state, c_prev, v_first
            )

            state[f"rwkv_x_{i}"] = x_prev
            state[f"rwkv_s_{i}"] = time_state
            state[f"rwkv_c_{i}"] = c_prev
        return x_t, state

    def forward_kernel_with_state(self, x, s0_list, reset_mask):
        if x.dtype != self.blocks[0].ln1.weight.dtype:
            raise RuntimeError("RWKV7 kernel input dtype mismatch with parameters.")
        v_first = None
        rwkv_cache = {}
        for i, block in enumerate(self.blocks):
            x, v_first, x_in, s_last, s_t, c_in = block.forward_kernel_with_state(
                x, v_first, s0_list[i], reset_mask
            )
            rwkv_cache[f"rwkv_x_{i}"] = x_in
            rwkv_cache[f"rwkv_s_{i}"] = s_t
            rwkv_cache[f"rwkv_c_{i}"] = c_in
        return x, rwkv_cache


class RWKV7StateManager:
    # Single source of truth for RWKV7 state init/reset/cache (no parallel state systems).
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

    def segment_starts(self, is_first_row, T):
        if is_first_row is None:
            return [0]
        starts = torch.nonzero(is_first_row > 0.5, as_tuple=False).view(-1).tolist()
        if 0 not in starts:
            starts = [0] + starts
        return sorted(set([s for s in starts if s < T]))

    def init_cache(self, batch_size, T, device, dtype):
        cache = {}
        for i in range(self.n_layer):
            cache[f"rwkv_x_{i}"] = torch.empty(batch_size, T, self.n_embd, device=device, dtype=dtype)
            cache[f"rwkv_c_{i}"] = torch.empty(batch_size, T, self.n_embd, device=device, dtype=dtype)
            cache[f"rwkv_s_{i}"] = torch.empty(
                batch_size, T, self.n_head, self.head_size, self.head_size, device=device, dtype=torch.float32
            )
        return cache
