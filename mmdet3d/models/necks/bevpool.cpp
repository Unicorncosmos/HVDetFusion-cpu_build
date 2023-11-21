// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
#include <torch/torch.h>

/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b, d, h, w, c]
*/
void bev_pool_v2_cpu(int c, int n_intervals, const float* depth, const float* feat,
                     const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
                     const int* interval_starts, const int* interval_lengths, float* out) {
  for (int index = 0; index < n_intervals; ++index) {
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];

    for (int cur_c = 0; cur_c < c; ++cur_c) {
      float psum = 0.0;
      for (int i = 0; i < interval_length; ++i) {
        const float* cur_depth = depth + ranks_depth[interval_start + i];
        const float* cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
        psum += *cur_feat * *cur_depth;
      }

      const int* cur_rank = ranks_bev + interval_start;
      float* cur_out = out + *cur_rank * c + cur_c;
      *cur_out = psum;
    }
  }
}

/*
  Function: pillar pooling backward
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    out_grad         : gradient of the BEV fmap from top, FloatTensor[b, d, h, w, c]
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    depth_grad       : gradient of the depth fmap, FloatTensor
    feat_grad        : gradient of the feature fmap, FloatTensor
*/
void bev_pool_v2_grad_cpu(int c, int n_intervals, const float* out_grad,
                          const float* depth, const float* feat,
                          const int* ranks_depth, const int* ranks_feat,
                          const int* ranks_bev, const int* interval_starts,
                          const int* interval_lengths, float* depth_grad, float* feat_grad) {
  for (int idx = 0; idx < n_intervals; ++idx) {
    int interval_start = interval_starts[idx];
    int interval_length = interval_lengths[idx];

    for (int i = 0; i < interval_length; ++i) {
      const int* cur_rank = ranks_bev + interval_start + i;
      const float* cur_out_grad_start = out_grad + *cur_rank * c;
      const float* cur_feat_start = feat + ranks_feat[interval_start + i] * c;

      float grad_sum = 0.0;
      for (int cur_c = 0; cur_c < c; ++cur_c) {
        const float* cur_out_grad = cur_out_grad_start + cur_c;
        const float* cur_feat = cur_feat_start + cur_c;
        grad_sum += *cur_out_grad * *cur_feat;
      }

      float* cur_depth_grad = depth_grad + ranks_depth[interval_start + i];
      *cur_depth_grad = grad_sum;
    }

    for (int cur_c = 0; cur_c < c; ++cur_c) {
      float grad_sum = 0.0;
      for (int i = 0; i < interval_length; ++i) {
        const int* cur_rank = ranks_bev + interval_start + i;
        const float* cur_out_grad = out_grad + *cur_rank * c + cur_c;

        const float* cur_depth = depth + ranks_depth[interval_start + i];
        grad_sum += *cur_out_grad * *cur_depth;
      }

      float* cur_feat_grad = feat_grad + ranks_feat[interval_start] * c + cur_c;
      *cur_feat_grad = grad_sum;
    }
  }
}

// Function declarations
void bev_pool_v2_cpu(int c, int n_intervals, const float* depth, const float* feat,
                     const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
                     const int* interval_starts, const int* interval_lengths, float* out);

void bev_pool_v2_grad_cpu(int c, int n_intervals, const float* out_grad,
                          const float* depth, const float* feat, const int* ranks_depth,
                          const int* ranks_feat, const int* ranks_bev,
                          const int* interval_starts, const int* interval_lengths,
                          float* depth_grad, float* feat_grad);

/*
  Function: pillar pooling (forward, cpu)
  Args:
    depth            : input depth, FloatTensor[n, d, h, w]
    feat             : input features, FloatTensor[n, h, w, c]
    out              : output features, FloatTensor[b, c, h_out, w_out]
    ranks_depth      : depth index of points, IntTensor[n_points]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
*/
void bev_pool_v2_forward_cpu(
  const at::Tensor _depth,
  const at::Tensor _feat,
  at::Tensor _out,
  const at::Tensor _ranks_depth,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts
) {
  int c = _feat.size(3);
  int n_intervals = _interval_lengths.size(0);
  
  const float* depth = _depth.data_ptr<float>();
  const float* feat = _feat.data_ptr<float>();
  const int* ranks_depth = _ranks_depth.data_ptr<int>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();

  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  float* out = _out.data_ptr<float>();
  bev_pool_v2_cpu(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}

/*
  Function: pillar pooling (backward, cpu)
  Args:
    out_grad         : grad of output bev feature, FloatTensor[b, c, h_out, w_out]
    depth_grad       : grad of input depth, FloatTensor[n, d, h, w]
    feat_grad        : grad of input feature, FloatTensor[n, h, w, c]
    depth            : input depth, FloatTensor[n, d, h, w]
    feat             : input features, FloatTensor[n, h, w, c]
    ranks_depth      : depth index of points, IntTensor[n_points]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
*/
void bev_pool_v2_backward_cpu(
  const at::Tensor _out_grad,
  at::Tensor _depth_grad,
  at::Tensor _feat_grad,
  const at::Tensor _depth,
  const at::Tensor _feat,
  const at::Tensor _ranks_depth,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts
) {
  int c = _out_grad.size(3);
  int n_intervals = _interval_lengths.size(0);

  const float* out_grad = _out_grad.data_ptr<float>();
  float* depth_grad = _depth_grad.data_ptr<float>();
  float* feat_grad = _feat_grad.data_ptr<float>();
  const float* depth = _depth.data_ptr<float>();
  const float* feat = _feat.data_ptr<float>();
  const int* ranks_depth = _ranks_depth.data_ptr<int>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  bev_pool_v2_grad_cpu(
    c, n_intervals, out_grad, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, depth_grad, feat_grad
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_pool_v2_forward", &bev_pool_v2_forward_cpu,
        "bev_pool_v2_forward_cpu");
  m.def("bev_pool_v2_backward", &bev_pool_v2_backward_cpu,
        "bev_pool_v2_backward_cpu");
}
