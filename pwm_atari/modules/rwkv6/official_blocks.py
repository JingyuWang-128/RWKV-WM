import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from modules.rwkv_cuda_env import configure_cuda_toolchain


try:
    _rwkv_testing = os.environ["RWKV_MY_TESTING"]
except KeyError:
    os.environ["RWKV_MY_TESTING"] = ""
    _rwkv_testing = ""


MyModule = nn.Module
MyFunction = lambda f: f

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"]) if "RWKV_HEAD_SIZE" in os.environ else None
_WKV6STREAM = None
_WKV6STREAM_HEAD_SIZE = None


def _use_rwkv6_cuda_fp32_diag():
    return os.environ.get("RWKV6_CUDA_FP32_DIAG", "0") == "1"


def rwkv6_stream_reference(r, k, v, w, u, s0, reset_mask, return_state_traj=True):
    B, T, C = r.shape
    H, N = u.shape
    assert C == H * N, (C, H, N)

    r_f = r.float().view(B, T, H, N)
    k_f = k.float().view(B, T, H, N)
    v_f = v.float().view(B, T, H, N)
    w_f = torch.exp(-torch.exp(w.float().view(B, T, H, N)))
    u_f = u.float().view(1, H, 1, N)
    state = s0.float().clone()

    ys = []
    s_traj = [] if return_state_traj else None
    for t in range(T):
        if reset_mask is not None:
            reset_t = reset_mask[:, t].view(B, 1, 1, 1)
            state = torch.where(reset_t, torch.zeros_like(state), state)
        k_t = k_f[:, t]
        v_t = v_f[:, t]
        r_t = r_f[:, t]
        w_t = w_f[:, t]
        kv = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        y = (((kv * u_f) + state) * r_t.unsqueeze(-2)).sum(dim=-1)
        state = state * w_t.unsqueeze(-2) + kv
        ys.append(y.reshape(B, C))
        if return_state_traj:
            s_traj.append(state.clone())

    y = torch.stack(ys, dim=1).to(r.dtype)
    s_last = state.to(s0.dtype)
    if return_state_traj:
        s_traj = torch.stack(s_traj, dim=1).to(s0.dtype)
    return y, s_last, s_traj


def _load_wkv6stream(head_size):
    global _WKV6STREAM, _WKV6STREAM_HEAD_SIZE
    if _WKV6STREAM is not None:
        if _WKV6STREAM_HEAD_SIZE != head_size:
            raise RuntimeError(
                f"RWKV6 kernel already initialized with head_size={_WKV6STREAM_HEAD_SIZE}, "
                f"requested head_size={head_size}."
            )
        return _WKV6STREAM

    include_candidates = configure_cuda_toolchain()

    root = Path(__file__).resolve().parent / "cuda"
    flags = [
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas",
        "-O3",
        "--extra-device-vectorization",
        "-U_FORTIFY_SOURCE",
        f"-D_N_={int(head_size)}",
    ]
    _WKV6STREAM = load(
        name=f"wkv6stream_h{int(head_size)}",
        sources=[str(root / "wkv6stream_op.cpp"), str(root / "wkv6stream_cuda.cu")],
        verbose=True,
        extra_include_paths=include_candidates,
        extra_cflags=["-U_FORTIFY_SOURCE"],
        extra_cuda_cflags=flags,
    )
    _WKV6STREAM_HEAD_SIZE = head_size
    return _WKV6STREAM


class _RWKV6StreamFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w, u, s0, reset_mask, head_size, return_state_traj):
        ctx.head_size = int(head_size)
        ctx.return_state_traj = bool(return_state_traj)
        ctx.orig_dtypes = (r.dtype, k.dtype, v.dtype, w.dtype, u.dtype, s0.dtype)
        ctx.use_reference_cuda = False

        B, T, C = r.shape
        H = C // int(head_size)
        if r.is_cuda and not _use_rwkv6_cuda_fp32_diag():
            ext = _load_wkv6stream(int(head_size))
            r_fp32 = r.contiguous().to(torch.float32)
            k_fp32 = k.contiguous().to(torch.float32)
            v_fp32 = v.contiguous().to(torch.float32)
            w_fp32 = w.contiguous().to(torch.float32)
            u_fp32 = u.contiguous().to(torch.float32)
            s0_fp32 = s0.contiguous().to(torch.float32)
            reset_mask = reset_mask.contiguous().to(torch.bool)
            y_fp32 = torch.empty_like(r_fp32)
            s_last_fp32 = torch.empty((B, H, head_size, head_size), device=r.device, dtype=torch.float32)
            s_traj_fp32 = torch.empty((B, T, H, head_size, head_size), device=r.device, dtype=torch.float32)
            ext.forward(B, T, C, H, r_fp32, k_fp32, v_fp32, w_fp32, u_fp32, s0_fp32, reset_mask, s_last_fp32, s_traj_fp32, y_fp32)
            ctx.save_for_backward(r_fp32, k_fp32, v_fp32, w_fp32, u_fp32, s0_fp32, reset_mask, s_traj_fp32)
            y = y_fp32.to(r.dtype)
            s_last = s_last_fp32
            s_traj = s_traj_fp32
        else:
            ctx.use_reference_cuda = bool(r.is_cuda)
            ctx.save_for_backward(r, k, v, w, u, s0, reset_mask, torch.empty(0, device=r.device, dtype=r.dtype))
            y, s_last, s_traj = rwkv6_stream_reference(r, k, v, w, u, s0, reset_mask, return_state_traj=True)
        if not return_state_traj:
            s_traj = s_traj[:, :0]
        return y, s_last, s_traj

    @staticmethod
    def backward(ctx, gy, gs_last, gs_traj):
        r, k, v, w, u, s0, reset_mask, s_traj = ctx.saved_tensors
        if r.is_cuda and not ctx.use_reference_cuda:
            ext = _load_wkv6stream(ctx.head_size)
            gy_fp32 = gy.contiguous().to(torch.float32)
            gs_last_fp32 = gs_last.contiguous().to(torch.float32)
            if gs_traj.numel():
                gs_traj_fp32 = gs_traj.contiguous().to(torch.float32)
            else:
                gs_traj_fp32 = torch.zeros_like(s_traj)
            gr = torch.empty_like(r, dtype=torch.float32)
            gk = torch.empty_like(k, dtype=torch.float32)
            gv = torch.empty_like(v, dtype=torch.float32)
            gw = torch.empty_like(w, dtype=torch.float32)
            gu_batch = torch.empty((r.shape[0], r.shape[2] // ctx.head_size, ctx.head_size), device=r.device, dtype=torch.float32)
            gs0 = torch.empty_like(s0, dtype=torch.float32)
            ext.backward(
                r.shape[0],
                r.shape[1],
                r.shape[2],
                r.shape[2] // ctx.head_size,
                r,
                k,
                v,
                w,
                u,
                s0,
                reset_mask,
                s_traj,
                gy_fp32,
                gs_last_fp32,
                gs_traj_fp32,
                gr,
                gk,
                gv,
                gw,
                gu_batch,
                gs0,
            )
            gu = gu_batch.sum(dim=0).to(ctx.orig_dtypes[4])
            return (
                gr.to(ctx.orig_dtypes[0]),
                gk.to(ctx.orig_dtypes[1]),
                gv.to(ctx.orig_dtypes[2]),
                gw.to(ctx.orig_dtypes[3]),
                gu,
                gs0.to(ctx.orig_dtypes[5]),
                None,
                None,
                None,
            )
        with torch.enable_grad():
            named_inputs = [
                ("r", r.detach().requires_grad_(r.requires_grad)),
                ("k", k.detach().requires_grad_(k.requires_grad)),
                ("v", v.detach().requires_grad_(v.requires_grad)),
                ("w", w.detach().requires_grad_(w.requires_grad)),
                ("u", u.detach().requires_grad_(u.requires_grad)),
                ("s0", s0.detach().requires_grad_(s0.requires_grad)),
            ]
            grad_input_names = [name for name, tensor in named_inputs if tensor.requires_grad]
            grad_inputs = [tensor for _, tensor in named_inputs if tensor.requires_grad]
            input_map = {name: tensor for name, tensor in named_inputs}
            y, s_last, s_traj_ref = rwkv6_stream_reference(
                input_map["r"],
                input_map["k"],
                input_map["v"],
                input_map["w"],
                input_map["u"],
                input_map["s0"],
                reset_mask,
                return_state_traj=True,
            )
            targets = [y, s_last]
            outputs = [gy, gs_last]
            if gs_traj.numel():
                targets.append(s_traj_ref)
                outputs.append(gs_traj)
            if grad_inputs:
                computed_grads = torch.autograd.grad(
                    targets,
                    grad_inputs,
                    grad_outputs=outputs,
                    allow_unused=True,
                )
                grad_map = dict(zip(grad_input_names, computed_grads))
            else:
                grad_map = {}
        grads = tuple(grad_map.get(name) for name, _ in named_inputs)
        return (*grads, None, None, None)

def RUN_CUDA_RWKV6_stream(r, k, v, w, u, s0, reset_mask, write_states=True):
    return _RWKV6StreamFn.apply(r, k, v, w, u, s0, reset_mask, HEAD_SIZE, write_states)


def RUN_CUDA_RWKV6(r, k, v, w, u):
    B, T, C = r.shape
    H = C // HEAD_SIZE
    s0 = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device=r.device, dtype=torch.float32)
    reset_mask = torch.zeros(B, T, device=r.device, dtype=torch.bool)
    y, _, _ = RUN_CUDA_RWKV6_stream(r, k, v, w, u, s0, reset_mask, False)
    return y


class RWKV_Tmix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(args.n_layer - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            d_mix_lora = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, d_mix_lora * 5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, d_mix_lora, args.n_embd).uniform_(-0.01, 0.01))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / max(args.dim_att - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

            d_decay_lora = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, d_decay_lora))
            self.time_decay_w2 = nn.Parameter(torch.zeros(d_decay_lora, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / max(args.dim_att - 1, 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        # Match the official x060 implementation's default head_size_divisor=8.
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=64e-5)

    def _project(self, x, xx):
        B, T, _ = x.size()
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))
        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww
        return r, k, v, w, g

    @MyFunction
    def forward(self, x):
        B, T, C = x.size()
        xx = self.time_shift(x) - x
        r, k, v, w, g = self._project(x, xx)
        x = RUN_CUDA_RWKV6(r, k, v, w, u=self.time_faaaa)
        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward_with_state(self, x, s0, reset_mask, write_states=True):
        B, T, C = x.size()
        x_prev = self.time_shift(x)
        if reset_mask is not None:
            x_prev = x_prev * (1.0 - reset_mask.view(B, T, 1).to(x_prev.dtype))
        xx = x_prev - x
        r, k, v, w, g = self._project(x, xx)
        x, s_last, s_t = RUN_CUDA_RWKV6_stream(r, k, v, w, self.time_faaaa, s0, reset_mask, write_states)
        x = x.view(B * T, C).to(self.ln_x.weight.dtype)
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x, s_last, s_t


class RWKV_CMix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**3))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**3))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r
        k = torch.relu(self.key(xk)) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

    def forward_with_reset(self, x, reset_mask):
        B, T, _ = x.size()
        x_prev = self.time_shift(x)
        if reset_mask is not None:
            x_prev = x_prev * (1.0 - reset_mask.view(B, T, 1).to(x_prev.dtype))
        xx = x_prev - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r
        k = torch.relu(self.key(xk)) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
