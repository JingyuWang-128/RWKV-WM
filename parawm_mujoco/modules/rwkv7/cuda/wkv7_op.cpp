#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cstdint>
using bf = __nv_bfloat16;

void cuda_forward_ext(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, float* s0, const uint8_t* reset, bf*y, float*s, float*sa, float* s_last, float* s_t, int write_states);
void cuda_forward_tail(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, float* s0, const uint8_t* reset, bf*y, float*sa, float* s_last, float* s_t, int write_states);

void forward_ext(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &s0, torch::Tensor &reset,
        torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa, torch::Tensor &s_last, torch::Tensor &s_t, int64_t write_states) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward_ext(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(),
            (float*)s0.data_ptr(), (const uint8_t*)reset.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)s_last.data_ptr(), (float*)s_t.data_ptr(), (int)write_states);
}

void forward_tail(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &s0, torch::Tensor &reset,
        torch::Tensor &y, torch::Tensor &sa, torch::Tensor &s_last, torch::Tensor &s_t, int64_t write_states) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward_tail(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(),
            (float*)s0.data_ptr(), (const uint8_t*)reset.data_ptr(), (bf*)y.data_ptr(), (float*)sa.data_ptr(), (float*)s_last.data_ptr(), (float*)s_t.data_ptr(), (int)write_states);
}

void cuda_backward_ext(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float* s0, const uint8_t* reset, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da);
void cuda_backward_tail(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float* s0, const uint8_t* reset, float* s_t, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da);

void backward_ext(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy, torch::Tensor &s0, torch::Tensor &reset,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward_ext(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(), 
            (float*)s0.data_ptr(), (const uint8_t*)reset.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr());
}

void backward_tail(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy, torch::Tensor &s0, torch::Tensor &reset,
        torch::Tensor &s_t, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward_tail(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(), 
            (float*)s0.data_ptr(), (const uint8_t*)reset.data_ptr(), (float*)s_t.data_ptr(), (float*)sa.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr());
}

TORCH_LIBRARY(wind_backstepping, m) {
    m.def("forward_ext(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor s0, Tensor reset, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa, Tensor(d!) s_last, Tensor(e!) s_t, int write_states) -> ()");
    m.def("forward_tail(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor s0, Tensor reset, Tensor(a!) y, Tensor(b!) sa, Tensor(c!) s_last, Tensor(d!) s_t, int write_states) -> ()");
    m.def("backward_ext(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s0, Tensor reset, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
    m.def("backward_tail(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s0, Tensor reset, Tensor s_t, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward_ext", &forward_ext);
    m.impl("forward_tail", &forward_tail);
    m.impl("backward_ext", &backward_ext);
    m.impl("backward_tail", &backward_tail);
}
