#include <assert.h>
#include <stdio.h>
#include "ATen/ATen.h"

__global__ void kernel_forward(
    const int B,
    const int T,
    const int C,
    const int H,
    const float *__restrict__ const _r,
    const float *__restrict__ const _k,
    const float *__restrict__ const _v,
    const float *__restrict__ const _w,
    const float *__restrict__ const _u,
    const float *__restrict__ const _s0,
    const bool *__restrict__ const _reset,
    float *__restrict__ const _s_last,
    float *__restrict__ const _s_traj,
    float *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    const float* u_ptr = _u + h * _N_;
    const float* s0_ptr = _s0 + b * H * _N_ * _N_ + h * _N_ * _N_ + i * _N_;
    float* s_last_ptr = _s_last + b * H * _N_ * _N_ + h * _N_ * _N_ + i * _N_;

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    float state[_N_];

    __syncthreads();
    u[i] = u_ptr[i];
    __syncthreads();
    for (int j = 0; j < _N_; j++) {
        state[j] = s0_ptr[j];
    }

    for (int t_local = 0; t_local < T; t_local++)
    {
        const int t = b * T * C + t_local * C + h * _N_ + i;
        if (_reset[b * T + t_local]) {
            #pragma unroll
            for (int j = 0; j < _N_; j++) {
                state[j] = 0.0f;
            }
        }

        __syncthreads();
        w[i] = __expf(-__expf(_w[t]));
        r[i] = _r[t];
        k[i] = _k[t];
        __syncthreads();

        const float v = _v[t];
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j += 4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }

        _y[t] = y;
        float* s_traj_ptr = _s_traj + b * T * H * _N_ * _N_ + t_local * H * _N_ * _N_ + h * _N_ * _N_ + i * _N_;
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            s_traj_ptr[j] = state[j];
        }
    }

    #pragma unroll
    for (int j = 0; j < _N_; j++) {
        s_last_ptr[j] = state[j];
    }
}

void cuda_forward(
    int B,
    int T,
    int C,
    int H,
    float *r,
    float *k,
    float *v,
    float *w,
    float *u,
    float *s0,
    bool *reset,
    float *s_last,
    float *s_traj,
    float *y)
{
    assert(H * _N_ == C);
    assert(_N_ % 4 == 0);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, s0, reset, s_last, s_traj, y);
}

__global__ void kernel_backward(
    const int B,
    const int T,
    const int C,
    const int H,
    const float *__restrict__ const _r,
    const float *__restrict__ const _k,
    const float *__restrict__ const _v,
    const float *__restrict__ const _w,
    const float *__restrict__ const _u,
    const float *__restrict__ const _s0,
    const bool *__restrict__ const _reset,
    const float *__restrict__ const _s_traj,
    const float *__restrict__ const _gy,
    const float *__restrict__ const _gs_last,
    const float *__restrict__ const _gs_traj,
    float *__restrict__ const _gr,
    float *__restrict__ const _gk,
    float *__restrict__ const _gv,
    float *__restrict__ const _gw,
    float *__restrict__ const _gu,
    float *__restrict__ const _gs0)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float r[_N_], k[_N_], v[_N_], u[_N_], wraw[_N_], wdecay[_N_], gy[_N_];
    __shared__ float accum_gr[_N_], accum_gk[_N_], accum_gu[_N_], accum_gw[_N_];

    float gstate[_N_];
    float gprev[_N_];
    float gu_total = 0.0f;

    const float *u_ptr = _u + h * _N_;
    const float *s0_row = _s0 + b * H * _N_ * _N_ + h * _N_ * _N_ + i * _N_;
    const float *gs_last_row = _gs_last + b * H * _N_ * _N_ + h * _N_ * _N_ + i * _N_;
    float *gs0_row = _gs0 + b * H * _N_ * _N_ + h * _N_ * _N_ + i * _N_;

    __syncthreads();
    u[i] = u_ptr[i];
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < _N_; j++) {
        gstate[j] = gs_last_row[j];
    }

    for (int t_local = T - 1; t_local >= 0; --t_local)
    {
        const int base = b * T * C + t_local * C + h * _N_;
        const bool rst = _reset[b * T + t_local];

        __syncthreads();
        accum_gr[i] = 0.0f;
        accum_gk[i] = 0.0f;
        accum_gu[i] = 0.0f;
        accum_gw[i] = 0.0f;
        r[i] = _r[base + i];
        k[i] = _k[base + i];
        v[i] = _v[base + i];
        wraw[i] = _w[base + i];
        wdecay[i] = __expf(-__expf(wraw[i]));
        gy[i] = _gy[base + i];
        __syncthreads();

        float gv_local = 0.0f;
        const float *gs_traj_row = _gs_traj + b * T * H * _N_ * _N_ + t_local * H * _N_ * _N_ + h * _N_ * _N_ + i * _N_;
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            const float prev = rst ? 0.0f : (
                t_local == 0
                    ? s0_row[j]
                    : _s_traj[
                        b * T * H * _N_ * _N_
                        + (t_local - 1) * H * _N_ * _N_
                        + h * _N_ * _N_
                        + i * _N_
                        + j
                    ]
            );
            const float gcur = gstate[j] + gs_traj_row[j];
            const float x = v[i] * k[j];
            atomicAdd(&accum_gr[j], gy[i] * (u[j] * x + prev));
            atomicAdd(&accum_gk[j], (gcur + gy[i] * r[j] * u[j]) * v[i]);
            atomicAdd(&accum_gu[j], gy[i] * r[j] * x);
            atomicAdd(&accum_gw[j], gcur * prev);
            gv_local += (gcur + gy[i] * r[j] * u[j]) * k[j];
            gprev[j] = rst ? 0.0f : (gcur * wdecay[j] + gy[i] * r[j]);
        }
        __syncthreads();

        _gr[base + i] = accum_gr[i];
        _gk[base + i] = accum_gk[i];
        _gv[base + i] = gv_local;
        _gw[base + i] = accum_gw[i] * (-__expf(wraw[i]) * wdecay[i]);
        gu_total += accum_gu[i];

        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            gstate[j] = gprev[j];
        }
    }

    _gu[b * H * _N_ + h * _N_ + i] = gu_total;
    #pragma unroll
    for (int j = 0; j < _N_; j++) {
        gs0_row[j] = gstate[j];
    }
}

void cuda_backward(
    int B,
    int T,
    int C,
    int H,
    float *r,
    float *k,
    float *v,
    float *w,
    float *u,
    float *s0,
    bool *reset,
    float *s_traj,
    float *gy,
    float *gs_last,
    float *gs_traj,
    float *gr,
    float *gk,
    float *gv,
    float *gw,
    float *gu,
    float *gs0)
{
    assert(H * _N_ == C);
    assert(_N_ <= 64);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(
        B, T, C, H, r, k, v, w, u, s0, reset, s_traj, gy, gs_last, gs_traj, gr, gk, gv, gw, gu, gs0
    );
}
