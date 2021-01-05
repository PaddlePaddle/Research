#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void KeQe(const T* v, const T* r, const T* s, T* y, const int total_num, const int k2, const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
      int fea_index = i % total_num;
      int sample_index = i / total_num;
      float sumsum = 0.0;
      for (int j = 0; j < k2; j++) {
        int cur_fea_index = int(r[sample_index*k2+j]) * total_num + fea_index; 
        sumsum += v[cur_fea_index] * s[sample_index*k2+j];
      }
      y[i] = sumsum;
  }
}

// 前向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class QeCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* V = ctx.Input<Tensor>("V");
    auto* R = ctx.Input<Tensor>("R");
    auto* S = ctx.Input<Tensor>("S");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto v = V->data<T>();
    auto r = R->data<T>();
    auto s = S->data<T>();
    auto y = out_t->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    const int total_num = V->dims()[0];
    const int k2 = R->dims()[1];


    int num = V->numel();
    int block = 512;
    int grid = (num + block - 1) / block;
    KeQe<T><<<grid, block, 0, dev_ctx.stream()>>>(v, r, s, y, total_num, k2, num);
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
// 注册前向的GPU Kernel
REGISTER_OP_CUDA_KERNEL(qe,
                        paddle::operators::QeCUDAKernel<CUDA, float>,
                        paddle::operators::QeCUDAKernel<CUDA, double>);
