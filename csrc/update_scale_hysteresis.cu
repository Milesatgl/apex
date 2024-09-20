#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void update_scale_hysteresis_cuda_kernel(float* current_scale,
                                                    int* growth_tracker,
                                                    int* hysteresis_tracker,
                                                    const float* found_inf,
                                                    double growth_factor,
                                                    double backoff_factor,
                                                    int growth_interval,
                                                    int hysteresis)
{
  if (*found_inf > 0) {
    *hysteresis_tracker -= 1;

    // Only reset the growth tracker when hysteresis is larger than zero
    if (*hysteresis_tracker > 0) {
      *growth_tracker = 0;
      return;
    }
  }

  if (*found_inf) {
    *current_scale = (*current_scale)*backoff_factor;
    *growth_tracker = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = (*growth_tracker) + 1;
    if (successful == growth_interval) {
      auto new_scale = static_cast<float>((*current_scale)*growth_factor);
      // Do not grow the scale past fp32 bounds to inf.
      if (isfinite(new_scale)) {
          *current_scale = new_scale;
      }
      *growth_tracker = 0;
    } else {
      *growth_tracker = successful;
    }
  }

  // Reset the hysteresis tracker if no infs are found
  if (*found_inf <= 0) {
    *hysteresis_tracker = hysteresis;
  }
}

at::Tensor update_scale_hysteresis_cuda(at::Tensor current_scale,
                                        at::Tensor growth_tracker,
                                        at::Tensor hysteresis_tracker,
                                        at::Tensor found_inf,
                                        const double growth_factor,
                                        const double backoff_factor,
                                        const int64_t growth_interval,
                                        const int hysteresis)
{
  // 使用 data_ptr 来获取指针，并通过 const关键字控制只读性
  float* current_scale_ptr = current_scale.data_ptr<float>();
  int* growth_tracker_ptr = growth_tracker.data_ptr<int>();
  int* hysteresis_tracker_ptr = hysteresis_tracker.data_ptr<int>();
  const float* found_inf_ptr = found_inf.data_ptr<float>(); // 这里我们手动加上 const

  update_scale_hysteresis_cuda_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
    current_scale_ptr,
    growth_tracker_ptr,
    hysteresis_tracker_ptr,
    found_inf_ptr,
    growth_factor,
    backoff_factor,
    growth_interval,
    hysteresis);

  AT_CUDA_CHECK(cudaGetLastError());

  return current_scale;
}

