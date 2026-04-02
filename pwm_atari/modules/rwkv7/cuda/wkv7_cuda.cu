#include <cuda_bf16.h>
#include <assert.h>
#include <stdint.h>

using bf = __nv_bfloat16;
__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return __float2bfloat16_rn(u); }

typedef bf * __restrict__ F_;
constexpr float W_SCALE = -0.6065306597f; // -exp(-0.5)

__global__ void forward_kernel_ext(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, const float* s0_, const uint8_t* reset_, bf* y_, float* s_, float* sa_, float* s_last_, float* s_t_, int write_states) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float state[C];
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    int s0_base = ((bb * H + hh) * C + i) * C;
#pragma unroll
    for (int j = 0; j < C; j++) {
        state[j] = s0_[s0_base + j];
    }

    for (int t = 0; t < T; t++) {
        if (reset_[bb * T + t]) {
#pragma unroll
            for (int j = 0; j < C; j++) {
                state[j] = s0_[s0_base + j];
            }
        }
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(W_SCALE / (1.0f + __expf(-to_float(w_[ind]))));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;

        float v = to_float(v_[ind]);
        float y = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v; // 状态更新核心公式
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
        }

        if (write_states) {
            int base_t = (((bb * T + t) * H + hh) * C + i) * C;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_t_[base_t + j] = state[j];
            }
        }
    }

    int base_last = ((bb * H + hh) * C + i) * C;
#pragma unroll
    for (int j = 0; j < C; j++) {
        s_last_[base_last + j] = state[j];
    }
}

__global__ void backward_kernel_ext(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, const float* s0_, const uint8_t* reset_, float * __restrict__ s_, float * __restrict__ sa_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float w_sig = 1.0f / (1.0f + __expf(-to_float(w_[ind])));
        w[i] = wi = __expf(W_SCALE * w_sig);
        k[i] = ki = to_float(k_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        v[i] = to_float(v_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
#pragma unroll
            for (int j = 0; j < C; j++) {
                stateT[j] = s_[base + j];
            }
        }

        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
        dq_[ind] = to_bf(dq);

        float iwi = 1.0f/wi;
        float prev[C];
#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
            if (reset_[bb * T + t]) {
                prev[j] = s0_[((bb * H + hh) * C + i) * C + j];
            } else {
                prev[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            }
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*prev[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = to_bf(W_SCALE * dw * wi * w_sig * (1.0f - w_sig));
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            da += prev[j]*dSb_shared[j];
        }
        da_[ind] = to_bf(da);

#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
        }

        if (reset_[bb * T + t]) {
#pragma unroll
            for (int j = 0; j < C; j++) {
                dstate[j] = 0;
                dstateT[j] = 0;
            }
        }
    }
}

__global__ void forward_kernel_tail(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, const float* s0_, const uint8_t* reset_, bf* y_, float* sa_, float* s_last_, float* s_t_, int write_states) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float state[C];
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    int s0_base = ((bb * H + hh) * C + i) * C;
#pragma unroll
    for (int j = 0; j < C; j++) {
        state[j] = s0_[s0_base + j];
    }

    for (int t = 0; t < T; t++) {
        if (reset_[bb * T + t]) {
#pragma unroll
            for (int j = 0; j < C; j++) {
                state[j] = s0_[s0_base + j];
            }
        }
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(W_SCALE / (1.0f + __expf(-to_float(w_[ind]))));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;

        float v = to_float(v_[ind]);
        float y = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        if (write_states) {
            int base_t = (((bb * T + t) * H + hh) * C + i) * C;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_t_[base_t + j] = state[j];
            }
        }
    }

    int base_last = ((bb * H + hh) * C + i) * C;
#pragma unroll
    for (int j = 0; j < C; j++) {
        s_last_[base_last + j] = state[j];
    }
}

__global__ void backward_kernel_tail(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, const float* s0_, const uint8_t* reset_, float * __restrict__ s_t_, float * __restrict__ sa_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float w_sig = 1.0f / (1.0f + __expf(-to_float(w_[ind])));
        w[i] = wi = __expf(W_SCALE * w_sig);
        k[i] = ki = to_float(k_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        v[i] = to_float(v_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        int base_t = (((bb * T + t) * H + hh) * C + i) * C;
#pragma unroll
        for (int j = 0; j < C; j++) {
            stateT[j] = s_t_[base_t + j];
        }

        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
        dq_[ind] = to_bf(dq);

        float iwi = 1.0f/wi;
        float prev[C];
#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
            if (reset_[bb * T + t]) {
                prev[j] = s0_[((bb * H + hh) * C + i) * C + j];
            } else {
                prev[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            }
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*prev[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = to_bf(W_SCALE * dw * wi * w_sig * (1.0f - w_sig));
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            da += prev[j]*dSb_shared[j];
        }
        da_[ind] = to_bf(da);

#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*iwi + ai * dSb_shared[j];
        }

        if (reset_[bb * T + t]) {
#pragma unroll
            for (int j = 0; j < C; j++) {
                dstate[j] = 0;
                dstateT[j] = 0;
            }
        }
    }
}

void cuda_forward_ext(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, float* s0, const uint8_t* reset, bf*y, float*s, float*sa, float* s_last, float* s_t, int write_states) {
    forward_kernel_ext<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,s0,reset,y,s,sa,s_last,s_t,write_states);
}
void cuda_backward_ext(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float* s0, const uint8_t* reset, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel_ext<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s0,reset,s,sa,dw,dq,dk,dv,dz,da);
}

void cuda_forward_tail(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, float* s0, const uint8_t* reset, bf*y, float*sa, float* s_last, float* s_t, int write_states) {
    forward_kernel_tail<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,s0,reset,y,sa,s_last,s_t,write_states);
}
void cuda_backward_tail(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float* s0, const uint8_t* reset, float* s_t, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da) {
    backward_kernel_tail<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s0,reset,s_t,sa,dw,dq,dk,dv,dz,da);
}
