########################################################################################################
# Minimal RWKV-7 official blocks and CUDA kernel loader (extracted from model_official_version.py)
########################################################################################################

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load


try:
    _rwkv_testing = os.environ["RWKV_MY_TESTING"]
except KeyError:
    os.environ["RWKV_MY_TESTING"] = ""
    _rwkv_testing = ""


MyModule = nn.Module
MyFunction = lambda f: f


HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"]) if "RWKV_HEAD_SIZE" in os.environ else None


def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    raise RuntimeError(
        "RUN_CUDA_RWKV7g not initialized. Set RWKV_MY_TESTING='x070' and RWKV_HEAD_SIZE to enable CUDA kernel."
    )


if "x070" in _rwkv_testing and HEAD_SIZE is not None:
    CHUNK_LEN = 16
    flags = [
        "-res-usage",
        f"-D_C_={HEAD_SIZE}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]
    _base = os.path.dirname(__file__)
    load(
        name="wind_backstepping",
        sources=[os.path.join(_base, "cuda", "wkv7_cuda.cu"), os.path.join(_base, "cuda", "wkv7_op.cpp")],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=flags,
    )

    class WindBacksteppingChunk(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b, s0, reset, write_states):
            B, T, H, C = w.shape
            assert T % CHUNK_LEN == 0
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
            assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
            assert s0.dtype == torch.float32 and s0.is_contiguous()
            assert reset.dtype == torch.uint8 and reset.is_contiguous()
            y = torch.empty_like(v)
            s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            s_last = torch.empty(B, H, C, C, dtype=torch.float32, device=w.device)
            s_t = torch.empty(B, T, H, C, C, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward_ext(w, q, k, v, z, b, s0, reset, y, s, sa, s_last, s_t, int(write_states))
            ctx.save_for_backward(w, q, k, v, z, b, s, sa, s0, reset)
            return y, s_last, s_t

        @staticmethod
        def backward(ctx, dy, ds_last, ds_t):
            assert all(i.dtype == torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w, q, k, v, z, b, s, sa, s0, reset = ctx.saved_tensors
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
            torch.ops.wind_backstepping.backward_ext(w, q, k, v, z, b, dy, s0, reset, s, sa, dw, dq, dk, dv, dz, db)
            return dw, dq, dk, dv, dz, db, None, None, None

    class WindBacksteppingTail(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b, s0, reset, write_states):
            B, T, H, C = w.shape
            assert T > 0 and T < CHUNK_LEN
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
            assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
            assert s0.dtype == torch.float32 and s0.is_contiguous()
            assert reset.dtype == torch.uint8 and reset.is_contiguous()
            y = torch.empty_like(v)
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            s_last = torch.empty(B, H, C, C, dtype=torch.float32, device=w.device)
            s_t = torch.empty(B, T, H, C, C, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward_tail(w, q, k, v, z, b, s0, reset, y, sa, s_last, s_t, int(write_states))
            ctx.save_for_backward(w, q, k, v, z, b, s_t, sa, s0, reset)
            return y, s_last, s_t

        @staticmethod
        def backward(ctx, dy, ds_last, ds_t):
            assert all(i.dtype == torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w, q, k, v, z, b, s_t, sa, s0, reset = ctx.saved_tensors
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
            torch.ops.wind_backstepping.backward_tail(w, q, k, v, z, b, dy, s0, reset, s_t, sa, dw, dq, dk, dv, dz, db)
            return dw, dq, dk, dv, dz, db, None, None, None

    def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
        B, T, HC = q.shape
        head = HEAD_SIZE
        q, w, k, v, a, b = [
            i.to(torch.bfloat16).view(B, T, HC // head, head).contiguous()
            for i in [q, w, k, v, a, b]
        ]
        s0 = torch.zeros(B, HC // head, head, head, dtype=torch.float32, device=q.device)
        reset = torch.zeros(B, T, device=q.device, dtype=torch.uint8)
        y, _, _ = WindBacksteppingChunk.apply(w, q, k, v, a, b, s0, reset, False)
        return y.view(B, T, HC)

    def RUN_CUDA_RWKV7g_stream(q, w, k, v, a, b, s0, reset_mask, write_states=True):
        B, T, HC = q.shape
        head = HEAD_SIZE
        q, w, k, v, a, b = [
            i.to(torch.bfloat16).view(B, T, HC // head, head).contiguous()
            for i in [q, w, k, v, a, b]
        ]
        if s0.dtype != torch.float32:
            s0 = s0.float()
        s0 = s0.contiguous()
        reset_mask = reset_mask.to(torch.uint8).contiguous()
        if T == 0:
            y = q.new_empty((B, 0, HC // head, head))
            s_last = s0
            s_t = s0.new_empty((B, 0, HC // head, head, head))
            return y.view(B, 0, HC), s_last, s_t

        full = (T // CHUNK_LEN) * CHUNK_LEN
        ys = []
        s_t_all = []
        s_last = s0
        if full > 0:
            y_full, s_last, s_t_full = WindBacksteppingChunk.apply(
                w[:, :full].contiguous(),
                q[:, :full].contiguous(),
                k[:, :full].contiguous(),
                v[:, :full].contiguous(),
                a[:, :full].contiguous(),
                b[:, :full].contiguous(),
                s_last,
                reset_mask[:, :full].contiguous(),
                write_states,
            )
            ys.append(y_full)
            s_t_all.append(s_t_full)
        tail = T - full
        if tail > 0:
            y_tail, s_last, s_t_tail = WindBacksteppingTail.apply(
                w[:, full:].contiguous(),
                q[:, full:].contiguous(),
                k[:, full:].contiguous(),
                v[:, full:].contiguous(),
                a[:, full:].contiguous(),
                b[:, full:].contiguous(),
                s_last,
                reset_mask[:, full:].contiguous(),
                True,
            )
            ys.append(y_tail)
            s_t_all.append(s_t_tail)
        y = torch.cat(ys, dim=1)
        s_t = torch.cat(s_t_all, dim=1) if write_states else s0.new_empty((B, 0, HC // head, head, head))
        return y.view(B, T, HC), s_last, s_t


class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1) if args.n_layer > 1 else 0
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5 if C > 1 else 0.0
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2) if N > 1 else 0.0
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3) if C > 1 else 0.0

            D_DECAY_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

            D_AAA_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

            D_MV_LORA = max(32, int(round((1.7 * (C**0.5)) / 32) * 32))
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

            D_GATE_LORA = max(32, int(round((5 * (C**0.5)) / 32) * 32))
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        # w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        s0 = torch.zeros(B, H, self.head_size, self.head_size, dtype=torch.float32, device=x.device)
        reset_mask = torch.zeros(B, T, device=x.device, dtype=torch.uint8)
        x, _, _ = RUN_CUDA_RWKV7g_stream(r, w, k, v, -kk, kk * a, s0, reset_mask, False)
        x = x.to(self.ln_x.weight.dtype)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first

    def forward_with_state(self, x, v_first, s0, reset_mask, write_states=True):
        B, T, C = x.size()
        H = self.n_head
        x_prev = self.time_shift(x)
        if reset_mask is not None:
            x_prev = x_prev * (1.0 - reset_mask.view(B, T, 1).to(x_prev.dtype))
        xx = x_prev - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        # w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x, s_last, s_t = RUN_CUDA_RWKV7g_stream(r, w, k, v, -kk, kk * a, s0, reset_mask, write_states)
        x = x.to(self.ln_x.weight.dtype)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first, s_last, s_t


class RWKV_CMix_x070(MyModule):
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
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5))
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

    def forward_with_reset(self, x, reset_mask):
        B, T, C = x.size()
        x_prev = self.time_shift(x)
        if reset_mask is not None:
            x_prev = x_prev * (1.0 - reset_mask.view(B, T, 1).to(x_prev.dtype))
        xx = x_prev - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)
