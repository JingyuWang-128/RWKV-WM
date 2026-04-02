#include <torch/extension.h>
#include "ATen/ATen.h"

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
    float *y);
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
    float *gs0);

void forward(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor &r,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &w,
    torch::Tensor &u,
    torch::Tensor &s0,
    torch::Tensor &reset,
    torch::Tensor &s_last,
    torch::Tensor &s_traj,
    torch::Tensor &y) {
    cuda_forward(
        B,
        T,
        C,
        H,
        r.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        w.data_ptr<float>(),
        u.data_ptr<float>(),
        s0.data_ptr<float>(),
        reset.data_ptr<bool>(),
        s_last.data_ptr<float>(),
        s_traj.data_ptr<float>(),
        y.data_ptr<float>());
}

void backward(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor &r,
    torch::Tensor &k,
    torch::Tensor &v,
    torch::Tensor &w,
    torch::Tensor &u,
    torch::Tensor &s0,
    torch::Tensor &reset,
    torch::Tensor &s_traj,
    torch::Tensor &gy,
    torch::Tensor &gs_last,
    torch::Tensor &gs_traj,
    torch::Tensor &gr,
    torch::Tensor &gk,
    torch::Tensor &gv,
    torch::Tensor &gw,
    torch::Tensor &gu,
    torch::Tensor &gs0) {
    cuda_backward(
        B,
        T,
        C,
        H,
        r.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        w.data_ptr<float>(),
        u.data_ptr<float>(),
        s0.data_ptr<float>(),
        reset.data_ptr<bool>(),
        s_traj.data_ptr<float>(),
        gy.data_ptr<float>(),
        gs_last.data_ptr<float>(),
        gs_traj.data_ptr<float>(),
        gr.data_ptr<float>(),
        gk.data_ptr<float>(),
        gv.data_ptr<float>(),
        gw.data_ptr<float>(),
        gu.data_ptr<float>(),
        gs0.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv6stream forward");
    m.def("backward", &backward, "wkv6stream backward");
}

TORCH_LIBRARY(wkv6stream, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
